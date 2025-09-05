#!/usr/bin/env python3
"""
unified_FITS_classification_parameter_agent_v02_3.py
=======================================
Unified FITS Classification + Parameter Agent - COMPLETE FIXED VERSION v2.1
* ORCHESTRATOR COMPATIBLE - Fixed missing question_category attribute
* Enhanced error handling and validation
* Improved LLM prompts for better classification
* Complete compatibility with FITSAnalysisOrchestrator

Single agent that handles both intent classification and parameter extraction
Perfect for MVP and single developer maintenance
"""

import asyncio
import json
import logging
import re
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib

# LangChain imports
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain.chat_models import ChatOpenAI

from langchain.schema import HumanMessage, SystemMessage
from langchain_community.callbacks.manager import get_openai_callback

from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class UnifiedFITSResult:
    """
    Unified result containing both classification and parameters
    ✅ ORCHESTRATOR COMPATIBLE - includes all required attributes
    """
    # Classification fields
    primary_intent: str                                           # "analysis", "general", "mixed"
    analysis_types: List[str] = field(default_factory=list)      # ["statistics", "psd", "fitting"]
    routing_strategy: str = "analysis"                           # "analysis", "astrosage", "mixed"
    confidence: float = 0.0
    reasoning: str = ""
    
    # Mixed request handling
    is_mixed_request: bool = False
    question_context: Optional[str] = None                       # "before_analysis", "after_analysis", "parallel", "standalone"
    astrosage_required: bool = False
    
    # ========================================
    # ✅ ORCHESTRATOR COMPATIBILITY FIX
    # ========================================
    question_category: str = "unknown"                           # "astronomy", "physics", "data_analysis", "methods", "unknown"
    complexity_level: str = "intermediate"                       # "beginner", "intermediate", "advanced"
    
    # Parameter fields for each analysis type
    parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # {"psd": {"low_freq": 1e-5}}
    parameter_confidence: Dict[str, float] = field(default_factory=dict)  # {"psd": 0.9}
    parameter_source: Dict[str, str] = field(default_factory=dict)       # {"psd": "user_specified"}
    
    # Workflow and guidance
    suggested_workflow: List[str] = field(default_factory=list)
    parameter_explanations: Dict[str, str] = field(default_factory=dict)
    potential_issues: List[str] = field(default_factory=list)
    
    # Metadata
    processing_time: float = 0.0
    tokens_used: int = 0
    cost_estimate: float = 0.0
    model_used: str = ""
    
    # Context enhancement
    requires_file: bool = False
    file_available: bool = False


class UnifiedFITSClassificationAgent:
    """
    Unified FITS Classification + Parameter Agent v2.1
    ORCHESTRATOR COMPATIBLE VERSION
    
    Single agent that handles both:
    1. Intent classification (analysis/general/mixed)
    2. Parameter extraction for FITS analysis
    3. Question categorization for AstroSage routing
    
    Perfect for MVP and single developer projects.
    """
    
    def __init__(self, 
                 model_name: str = "gpt-3.5-turbo", 
                 temperature: float = 0.1,
                 max_tokens: int = 1500):
        self.name = "UnifiedFITSAgent"
        self.logger = logging.getLogger(f"agent.{self.name}")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=30
        )
        
        # Parameter schemas for validation
        self.parameter_schemas = self._build_parameter_schemas()
        
        # Build comprehensive system prompt with question categorization
        self.system_prompt = self._build_comprehensive_system_prompt()
        
        # Simple cache for frequently used requests
        self.cache: Dict[str, UnifiedFITSResult] = {}
        self.cache_ttl = 1800  # 30 minutes
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "intent_distribution": {"analysis": 0, "general": 0, "mixed": 0},
            "question_categories": {"astronomy": 0, "physics": 0, "data_analysis": 0, "methods": 0, "unknown": 0},
            "parameter_extractions": 0,
            "avg_processing_time": 0.0
        }
        
        self.logger.info(f"Unified FITS Agent v2.1 initialized with {model_name}")
    
    def _build_parameter_schemas(self) -> Dict[str, Dict]:
        """Build comprehensive parameter schemas for all analysis types"""
        return {
            "statistics": {
                "parameters": {
                    "metrics": {
                        "type": "list",
                        "default": ["mean", "median", "std", "min", "max", "count"],
                        "options": ["mean", "median", "std", "min", "max", "count", "percentile_25", "percentile_75", "percentile_90"],
                        "description": "Statistical metrics to calculate"
                    }
                },
                "defaults": {"metrics": ["mean", "median", "std", "min", "max", "count"]},
                "validation": {
                    "metrics": "Must be subset of available metrics"
                }
            },
            
            "psd": {
                "parameters": {
                    "low_freq": {
                        "type": "float",
                        "default": 1e-5,
                        "range": [1e-6, 0.1],
                        "description": "Minimum frequency for PSD computation (Hz)"
                    },
                    "high_freq": {
                        "type": "float", 
                        "default": 0.05,
                        "range": [0.001, 0.5],
                        "description": "Maximum frequency for PSD computation (Hz)"
                    },
                    "bins": {
                        "type": "int",
                        "default": 3500,
                        "range": [100, 10000],
                        "description": "Number of frequency bins for PSD"
                    }
                },
                "defaults": {"low_freq": 1e-5, "high_freq": 0.05, "bins": 3500},
                "validation": {
                    "low_freq": "Must be > 0 and < high_freq",
                    "high_freq": "Must be > low_freq and <= 0.5",
                    "bins": "Must be integer between 100-10000"
                }
            },
            
            "fitting_power_law": {
                "parameters": {
                    "low_freq": {"type": "float", "default": 1e-5, "range": [1e-6, 0.1]},
                    "high_freq": {"type": "float", "default": 0.05, "range": [0.001, 0.5]},
                    "bins": {"type": "int", "default": 3500, "range": [100, 10000]},
                    "noise_bound_percent": {"type": "float", "default": 0.7, "range": [0.1, 1.0]},
                    "A0": {"type": "float", "default": 1.0, "range": [0.1, 100.0]},
                    "b0": {"type": "float", "default": 1.0, "range": [0.1, 3.0]},
                    "A_min": {"type": "float", "default": 0.0, "range": [0.0, 1e10]},
                    "A_max": {"type": "float", "default": 1e38, "range": [1.0, 1e38]},
                    "b_min": {"type": "float", "default": 0.1, "range": [0.01, 2.0]},
                    "b_max": {"type": "float", "default": 3.0, "range": [1.0, 5.0]},
                    "maxfev": {"type": "int", "default": 1000000, "range": [1000, 10000000]}
                },
                "defaults": {
                    "low_freq": 1e-5, "high_freq": 0.05, "bins": 3500,
                    "noise_bound_percent": 0.7, "A0": 1.0, "b0": 1.0,
                    "A_min": 0.0, "A_max": 1e38, "b_min": 0.1, "b_max": 3.0,
                    "maxfev": 1000000
                }
            },
            
            "fitting_bending_power_law": {
                "parameters": {
                    "low_freq": {"type": "float", "default": 1e-5, "range": [1e-6, 0.1]},
                    "high_freq": {"type": "float", "default": 0.05, "range": [0.001, 0.5]},
                    "bins": {"type": "int", "default": 3500, "range": [100, 10000]},
                    "noise_bound_percent": {"type": "float", "default": 0.7, "range": [0.1, 1.0]},
                    "A0": {"type": "float", "default": 10.0, "range": [0.1, 1000.0]},
                    "fb0": {"type": "float", "default": 0.01, "range": [1e-5, 0.1]},
                    "sh0": {"type": "float", "default": 1.0, "range": [0.1, 5.0]},
                    "A_min": {"type": "float", "default": 0.0, "range": [0.0, 1e10]},
                    "A_max": {"type": "float", "default": 1e38, "range": [1.0, 1e38]},
                    "fb_min": {"type": "float", "default": 2e-5, "range": [1e-6, 0.01]},
                    "fb_max": {"type": "float", "default": 0.05, "range": [0.001, 0.1]},
                    "sh_min": {"type": "float", "default": 0.3, "range": [0.1, 2.0]},
                    "sh_max": {"type": "float", "default": 3.0, "range": [1.0, 5.0]},
                    "maxfev": {"type": "int", "default": 1000000, "range": [1000, 10000000]}
                },
                "defaults": {
                    "low_freq": 1e-5, "high_freq": 0.05, "bins": 3500,
                    "noise_bound_percent": 0.7, "A0": 10.0, "fb0": 0.01, "sh0": 1.0,
                    "A_min": 0.0, "A_max": 1e38, "fb_min": 2e-5, "fb_max": 0.05,
                    "sh_min": 0.3, "sh_max": 3.0, "maxfev": 1000000
                }
            }
        }
    
    def _build_comprehensive_system_prompt(self) -> str:
        """Build comprehensive system prompt for unified classification + parameters + question categorization"""
        """Build system prompt with enhanced shared parameter handling"""

        # Build parameter documentation
        param_docs = ""
        for analysis_type, schema in self.parameter_schemas.items():
            param_docs += f"\n{analysis_type.upper().replace('_', ' ')} PARAMETERS:\n"
            
            parameters = schema.get("parameters", {})
            defaults = schema.get("defaults", {})
            
            for param_name, param_info in parameters.items():
                description = param_info.get('description', f'{param_name} parameter')
                default_value = defaults.get(param_name, param_info.get('default', 'N/A'))
                param_docs += f"  - {param_name}: {description} (default: {default_value})\n"

        # Enhanced shared parameter documentation
        shared_param_docs = """
            🔄 CRITICAL: COMPREHENSIVE SHARED PARAMETER HANDLING

            PARAMETER OVERLAP MATRIX:
            ┌─────────────────────┬─────────┬─────┬───────────┬─────────────┐
            │ Parameter           │ Stats   │ PSD │ Power Law │ Bending Law │
            ├─────────────────────┼─────────┼─────┼───────────┼─────────────┤
            │ low_freq            │   ❌    │ ✅  │    ✅     │     ✅      │  TRIPLE OVERLAP
            │ high_freq           │   ❌    │ ✅  │    ✅     │     ✅      │  TRIPLE OVERLAP  
            │ bins                │   ❌    │ ✅  │    ✅     │     ✅      │  TRIPLE OVERLAP
            │ A0                  │   ❌    │ ❌  │    ✅     │     ✅      │ 🔴 SPECIAL SHARED
            │ noise_bound_percent │   ❌    │ ❌  │    ✅     │     ✅      │ 🔄 FITTING SHARED
            │ A_min, A_max        │   ❌    │ ❌  │    ✅     │     ✅      │ 🔄 FITTING SHARED
            │ maxfev              │   ❌    │ ❌  │    ✅     │     ✅      │ 🔄 FITTING SHARED
            └─────────────────────┴─────────┴─────┴───────────┴─────────────┘

            🎯 SHARED PARAMETER RESOLUTION RULES:

            1. **TRIPLE OVERLAP PARAMETERS** (PSD + Both Fitting):
                - "frequency range X to Y" → low_freq: X, high_freq: Y in ALL THREE
                - "X bins" → bins: X in ALL THREE  
                - "bins from X to Y" → bins: X in ALL THREE (use first value)

            2. **FITTING-ONLY SHARED PARAMETERS** (Both Fitting Models):
                - "noise bound X%" → noise_bound_percent: X in BOTH fitting models
                - "amplitude bounds X to Y" → A_min: X, A_max: Y in BOTH fitting models
                - "max iterations X" → maxfev: X in BOTH fitting models

            3. **SPECIAL SHARED PARAMETER - A0**:
                - Different defaults: power_law=1.0, bending=10.0
                - "amplitude X" without model specification → apply to BOTH but explain
                - "power law amplitude X" → apply only to power law
                - "bending amplitude X" → apply only to bending power law
                - No amplitude specified → use model-specific defaults

            4. **CONFLICT RESOLUTION**:
                - Same parameter with different values → USE LAST MENTIONED
                - Ambiguous specification → USE CONTEXT or EXPLAIN
                - Model-specific override → APPLY TO SPECIFIC MODEL ONLY

            CRITICAL EXAMPLES:

            Example 1 - TRIPLE OVERLAP:
            Input: "Compute PSD and fit both power law models with frequency range 1e-4 to 0.1 Hz and 5000 bins"
            Response: Apply shared values to ALL THREE analysis types

            Example 2 - A0 SPECIAL HANDLING:
            Input: "Fit both models with amplitude 5.0"
            Response: Both models get A0=5.0, explain potential suboptimality

            Example 3 - CONFLICT RESOLUTION:
            Input: "Compute PSD with 4000 bins, then fit power law with 5000 bins"
            Response: Use 5000 bins for both (last mentioned), explain resolution

            GOLDEN RULE: "One specification, universal application" unless explicitly overridden
            """
        
        return f"""You are an expert AI agent specializing in FITS (Flexible Image Transport System) file analysis for astrophysical data.

            CORE MISSION:
            Provide unified classification, question categorization, and parameter extraction for astronomical time-series analysis requests.

            🔬 DOMAIN EXPERTISE:
            - FITS file analysis and astronomical data processing
            - Power Spectral Density (PSD) analysis for timing studies
            - Power law and bending power law model fitting
            - Statistical analysis of lightcurve data
            - X-ray astronomy, pulsar observations, neutron star studies

            TASK DESCRIPTION:
            You must provide BOTH classification and parameter extraction in a single response:

            1. **INTENT CLASSIFICATION**: Determine if the request is:
                - "analysis": Direct data analysis request
                - "general": Pure astronomy/physics questions
                - "mixed": Combination of questions and analysis

            2. **QUESTION CATEGORIZATION**: For general/mixed requests, categorize the question:
                - "astronomy": Celestial objects, stellar evolution, cosmic phenomena
                - "physics": Physical processes, laws, fundamental concepts
                - "data_analysis": Analysis methods, statistical concepts, PSD, fitting
                - "methods": Techniques, procedures, best practices
                - "unknown": Cannot categorize or mixed categories

            3. **PARAMETER EXTRACTION**: For analysis requests, extract specific parameters or provide intelligent defaults

            CLASSIFICATION RULES:

            **ANALYSIS REQUESTS** (routing_strategy: "analysis"):
            - Direct requests to analyze user's data WITHOUT explanatory questions
            - Keywords: "calculate", "compute", "fit", "analyze", "get statistics"
            - Examples: "Calculate mean and std", "Fit power law", "Compute PSD"
            - Pure data processing requests

            **GENERAL QUESTIONS** (routing_strategy: "astrosage"):
            - Pure astronomy/physics questions without data analysis
            - Keywords: "what", "why", "how", "explain", "tell me about"
            - Examples: "What is a neutron star?", "Explain stellar evolution"
            - No data processing involved

            **MIXED REQUESTS** (routing_strategy: "mixed"):
            - CRITICAL: Any request that combines data analysis AND explanation/interpretation
            - Patterns to detect as MIXED:
            * "Do X and explain Y" → mixed
            * "Calculate X and tell me what Y means" → mixed  
            * "What is X? Then do Y" → mixed
            * "I'm new to... do X and explain..." → mixed
            * "Do X, then explain what the results indicate" → mixed
            * Any analysis request that asks for interpretation or explanation
            - Examples: 
            * "What is PSD? Then compute it" → mixed, question_category: "data_analysis"
            * "Calculate statistics and explain what they mean" → mixed, question_category: "data_analysis"
            * "Compute PSD and explain what frequency peaks indicate" → mixed, question_category: "astronomy"
            * "I'm new to astronomy. Calculate statistics and explain them" → mixed, question_category: "astronomy"

            **KEY MIXED REQUEST INDICATORS:**
            - "explain what [results] mean/indicate"
            - "tell me what [results] show"
            - "interpret the [analysis results]"
            - "what do [analysis results] indicate about..."
            - "I'm new to..." followed by analysis request
            - Analysis + interpretation in same request

            ANALYSIS TYPES:
            - **statistics**: Basic statistical analysis (mean, median, std, etc.)
            - **psd**: Power Spectral Density computation
            - **fitting_power_law**: Simple power law model fitting
            - **fitting_bending_power_law**: Bending power law model fitting

            PARAMETER SCHEMAS:
            {param_docs}

            {shared_param_docs}

            QUESTION CATEGORIZATION EXAMPLES:
            - "What is a neutron star?" → question_category: "astronomy"
            - "Explain stellar evolution" → question_category: "astronomy" 
            - "How do black holes form?" → question_category: "astronomy"
            - "What causes X-ray emission?" → question_category: "physics"
            - "Explain quantum mechanics" → question_category: "physics"
            - "What is power spectral density?" → question_category: "data_analysis"
            - "How do I interpret PSD results?" → question_category: "data_analysis"
            - "What's the best way to fit models?" → question_category: "methods"
            - "How should I analyze timing data?" → question_category: "methods"

            RESPONSE FORMAT:
            Always respond with this exact JSON structure:
            {{
                "classification": {{
                    "primary_intent": "analysis|general|mixed",
                    "analysis_types": ["statistics", "psd", "fitting_power_law", "fitting_bending_power_law"],
                    "routing_strategy": "analysis|astrosage|mixed",
                    "confidence": 0.0-1.0,
                    "reasoning": "Clear explanation of classification decision",
                    "is_mixed_request": true|false,
                    "question_context": "before_analysis|after_analysis|parallel|standalone",
                    "astrosage_required": true|false,
                    "question_category": "astronomy|physics|data_analysis|methods|unknown",
                    "complexity_level": "beginner|intermediate|advanced"
                }},
                "parameters": {{
                    "statistics": {{"metrics": ["mean", "std"]}},
                    "psd": {{"low_freq": 1e-5, "high_freq": 0.05, "bins": 3500}},
                    "fitting_power_law": {{"A0": 1.0, "b0": 1.0, "noise_bound_percent": 0.7}},
                    "fitting_bending_power_law": {{"A0": 10.0, "fb0": 0.01, "sh0": 1.0}}
                }},
                "parameter_confidence": {{
                    "statistics": 0.9,
                    "psd": 0.8,
                    "fitting_power_law": 0.7
                }},
                "parameter_source": {{
                    "statistics": "user_specified|defaults_used|inferred",
                    "psd": "user_specified|defaults_used|inferred"
                }},
                "workflow": {{
                    "suggested_steps": ["load_fits", "compute_psd", "fit_model"],
                    "execution_pattern": "single|sequential|parallel",
                    "estimated_time": "30 seconds"
                }},
                "guidance": {{
                    "parameter_explanations": {{"bins": "Using 3500 bins for good frequency resolution"}},
                    "potential_issues": ["check_data_quality", "verify_frequency_range"],
                    "optimization_suggestions": ["consider_higher_bins_for_better_resolution"]
                }},
                "context": {{
                    "requires_file": true|false,
                    "complexity_level": "basic|intermediate|advanced",
                    "expertise_adaptation": "beginner|intermediate|expert"
                }}
            }}

            CRITICAL EXAMPLES:

            Example 1 - MIXED REQUEST:
            Input: "I'm new to X-ray astronomy. Calculate basic statistics for my neutron star data and explain what they mean"
            Classification: "mixed" (has both analysis AND explanation request)
            Routing: "mixed"
            Question Category: "astronomy"
            AstroSage Required: true

            Example 2 - MIXED REQUEST:
            Input: "Calculate statistics and compute PSD, then explain what the frequency peaks might indicate about the neutron star's rotation"
            Classification: "mixed" (analysis + interpretation request)
            Routing: "mixed"
            Question Category: "astronomy"
            AstroSage Required: true

            Example 3 - PURE ANALYSIS:
            Input: "Perform complete timing analysis: compute PSD with 4000 bins, then fit both power law and bending power law models"
            Classification: "analysis" (pure data processing, no explanation requested)
            Routing: "analysis"
            Question Category: "unknown" (no question asked)
            AstroSage Required: false

            Example 4 - PURE GENERAL:
            Input: "What causes neutron stars to emit X-rays?"
            Classification: "general" (pure question, no analysis)
            Routing: "astrosage"
            Question Category: "astronomy"
            AstroSage Required: true

            Example 5 - DATA ANALYSIS QUESTION:
            Input: "What is power spectral density and how do I interpret the results?"
            Classification: "general" (pure question)
            Routing: "astrosage"
            Question Category: "data_analysis"
            AstroSage Required: true

            PARAMETER EXTRACTION RULES:
            - "shape parameter X" → sh0: X (for bending power law)
            - "amplitude bounds from X to Y" → A_min: X, A_max: Y
            - "power index X" → b0: X (for power law)
            - "break frequency X" → fb0: X (for bending power law)
            - "frequency range X to Y" → low_freq: X, high_freq: Y
            - "X bins" → bins: X
            - Pay careful attention to parameter names and their correct mapping

            QUALITY STANDARDS:
            - Be precise with analysis type detection
            - Correctly categorize questions for optimal AstroSage routing
            - Only extract parameters explicitly mentioned or strongly implied
            - Provide intelligent defaults for missing parameters
            - Consider user expertise level for parameter suggestions
            - Include workflow guidance for complex analyses
            - Identify potential issues proactively
            - CORRECTLY identify mixed requests that need both analysis AND explanation"""
                
    async def process_request(self, 
                            user_input: str, 
                            context: Dict[str, Any] = None) -> UnifiedFITSResult:
        """
        Main method: Unified classification and parameter extraction with question categorization
        ✅ ORCHESTRATOR COMPATIBLE
        
        Args:
            user_input: User's natural language request
            context: Additional context (files, expertise, etc.)
            
        Returns:
            UnifiedFITSResult with classification, question categorization, and parameters
        """
        if context is None:
            context = {}
        
        start_time = datetime.now()
        self.stats["total_requests"] += 1
        
        # Clean up expired cache entries
        self._cleanup_cache()
        
        # Check cache first
        cache_key = self._generate_cache_key(user_input, context)
        if cache_key in self.cache:
            self.stats["cache_hits"] += 1
            cached_result = self.cache[cache_key]
            cached_result.processing_time = 0.001  # Almost instant
            self.logger.info(f"Cache hit for request: {user_input[:50]}...")
            return cached_result
        
        try:
            # Build comprehensive prompt
            full_prompt = self._build_unified_prompt(user_input, context)
            
            # Single LLM call for everything
            with get_openai_callback() as cb:
                messages = [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=full_prompt)
                ]
                
                response = await self.llm.agenerate([messages])
                raw_output = response.generations[0][0].text.strip()
                
                # Track usage
                tokens_used = cb.total_tokens if hasattr(cb, 'total_tokens') else 0
                cost = cb.total_cost if hasattr(cb, 'total_cost') else tokens_used * 0.000002
                
                self.stats["total_tokens"] += tokens_used
                self.stats["total_cost"] += cost
            
            # Parse unified response with question categorization
            result = self._parse_unified_response(raw_output)
            
            # Enhance with metadata
            processing_time = (datetime.now() - start_time).total_seconds()
            result.processing_time = processing_time
            result.tokens_used = tokens_used
            result.cost_estimate = cost
            result.model_used = self.llm.model_name
            
            # Apply context enhancements
            result = self._enhance_with_context(result, context)
            
            # Validate parameters
            result = self._validate_and_optimize_parameters(result)
            
            # Update statistics
            self.stats["intent_distribution"][result.primary_intent] += 1
            self.stats["question_categories"][result.question_category] += 1
            if result.parameters:
                self.stats["parameter_extractions"] += 1
            
            # Update average processing time
            total = self.stats["total_requests"]
            self.stats["avg_processing_time"] = (
                (self.stats["avg_processing_time"] * (total - 1) + processing_time) / total
            )
            
            # Cache the result
            self.cache[cache_key] = result
            self.cache_timestamps[cache_key] = datetime.now()
            
            self.logger.info(f"Unified processing completed: {result.primary_intent} → {result.analysis_types} "
                           f"(confidence: {result.confidence:.2f}, time: {processing_time:.3f}s)")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Unified processing failed: {str(e)}")
            return self._create_fallback_result(user_input, str(e))
    
    def _build_unified_prompt(self, user_input: str, context: Dict[str, Any]) -> str:
        """Build comprehensive prompt for unified processing with question categorization"""
        
        # Build context section
        context_info = []
        if context.get("has_uploaded_files"):
            context_info.append("✅ User has uploaded FITS files")
        else:
            context_info.append("❌ No FITS files uploaded yet")
        
        if context.get("user_expertise"):
            expertise = context["user_expertise"]
            context_info.append(f"👤 User expertise: {expertise}")
        
        if context.get("previous_analyses"):
            prev = ", ".join(context["previous_analyses"])
            context_info.append(f"📊 Previous analyses: {prev}")
        
        return f"""
            UNIFIED FITS ANALYSIS REQUEST

            USER INPUT: "{user_input}"

            CONTEXT INFORMATION:
            {chr(10).join(context_info) if context_info else "No additional context"}

            TASK: Provide comprehensive classification, question categorization, and parameter extraction in single response.

            ANALYSIS GUIDELINES:
            1. **Classification**: Determine primary intent and required analysis types
            2. **Question Categorization**: Classify questions for optimal AstroSage routing
            3. **Parameters**: Extract specific values mentioned or provide intelligent defaults
            4. **Workflow**: Suggest optimal analysis sequence
            5. **Guidance**: Provide helpful explanations and potential issue warnings

            PARAMETER EXTRACTION RULES:
            - Only extract parameters explicitly mentioned or strongly implied
            - Use domain expertise to suggest optimal defaults
            - Consider user expertise level for complexity
            - Validate parameter ranges and relationships
            - Provide explanations for non-obvious choices

            RESPOND WITH COMPLETE JSON (no truncation):
            """
    
    def _parse_unified_response(self, raw_output: str) -> UnifiedFITSResult:
        """Parse LLM JSON response into UnifiedFITSResult with question categorization"""
        try:
            # Clean and extract JSON
            cleaned_output = self._extract_json_from_response(raw_output)
            response_data = json.loads(cleaned_output)
            
            # Extract classification data
            classification = response_data.get("classification", {})
            parameters = response_data.get("parameters", {})
            param_confidence = response_data.get("parameter_confidence", {})
            param_source = response_data.get("parameter_source", {})
            workflow = response_data.get("workflow", {})
            guidance = response_data.get("guidance", {})
            context_data = response_data.get("context", {})
            
            # ===== ORCHESTRATOR COMPATIBILITY =====
            # Extract and validate question_category
            question_category = classification.get("question_category", "unknown")
            valid_categories = ["astronomy", "physics", "data_analysis", "methods", "unknown"]
            if question_category not in valid_categories:
                question_category = "unknown"
            
            # Extract and validate complexity_level
            complexity_level = classification.get("complexity_level", "intermediate")
            valid_levels = ["beginner", "intermediate", "advanced"]
            if complexity_level not in valid_levels:
                complexity_level = "intermediate"
            
            # Create unified result with all required attributes
            result = UnifiedFITSResult(
                # Classification fields
                primary_intent=classification.get("primary_intent", "analysis"),
                analysis_types=classification.get("analysis_types", []),
                routing_strategy=classification.get("routing_strategy", "analysis"),
                confidence=float(classification.get("confidence", 0.7)),
                reasoning=classification.get("reasoning", "Unified classification"),
                
                # Mixed request handling
                is_mixed_request=classification.get("is_mixed_request", False),
                question_context=classification.get("question_context"),
                astrosage_required=classification.get("astrosage_required", False),
                
                # ===== ORCHESTRATOR REQUIRED ATTRIBUTES =====
                question_category=question_category,
                complexity_level=complexity_level,
                
                # Parameter fields
                parameters=parameters,
                parameter_confidence=param_confidence,
                parameter_source=param_source,
                
                # Workflow and guidance
                suggested_workflow=workflow.get("suggested_steps", []),
                parameter_explanations=guidance.get("parameter_explanations", {}),
                potential_issues=guidance.get("potential_issues", []),
                
                # Context
                requires_file=context_data.get("requires_file", False)
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to parse unified response: {str(e)}")
            self.logger.error(f"Raw output: {raw_output[:500]}...")
            return self._create_fallback_parsing_result(raw_output)
    
    def _extract_json_from_response(self, text: str) -> str:
        """Extract JSON from various response formats"""
        text = text.strip()
        
        # Method 1: JSON code blocks
        json_pattern = r'```json\s*(.*?)\s*```'
        json_match = re.search(json_pattern, text, re.DOTALL | re.IGNORECASE)
        if json_match:
            return json_match.group(1).strip()
        
        # Method 2: Find JSON object
        brace_pattern = r'\{.*\}'
        brace_match = re.search(brace_pattern, text, re.DOTALL)
        if brace_match:
            return brace_match.group(0).strip()
        
        # Method 3: Assume entire text is JSON
        return text
    
    def _enhance_with_context(self, result: UnifiedFITSResult, context: Dict[str, Any]) -> UnifiedFITSResult:
        """Enhance result with context information"""
        
        # File availability
        if context.get("has_uploaded_files"):
            result.file_available = True
            if result.analysis_types:
                result.requires_file = True
        
        # User expertise adaptation
        user_expertise = context.get("user_expertise", "intermediate")
        if user_expertise in ["beginner", "intermediate", "advanced"]:
            result.complexity_level = user_expertise
        
        if user_expertise == "beginner" and result.primary_intent == "analysis":
            # Add educational guidance for beginners
            if not result.is_mixed_request:
                result.astrosage_required = True
                result.suggested_workflow.append("provide_beginner_explanation")
        
        return result
    
    def _validate_and_optimize_parameters(self, result: UnifiedFITSResult) -> UnifiedFITSResult:
        """Enhanced parameter validation with comprehensive shared parameter handling"""
        
        # Define shared parameter groups
        TRIPLE_OVERLAP_PARAMS = {"low_freq", "high_freq", "bins"}  # PSD + Both Fitting
        FITTING_SHARED_PARAMS = {"noise_bound_percent", "A_min", "A_max", "maxfev"}  # Both Fitting only
        SPECIAL_SHARED_PARAMS = {"A0"}  # Same name, different defaults
        
        # Step 1: Identify analysis types by category
        psd_types = [t for t in result.analysis_types if t == "psd"]
        fitting_types = [t for t in result.analysis_types if t.startswith("fitting_")]
        frequency_analysis_types = psd_types + fitting_types  # Types that use frequency parameters
        
        # Step 2: Handle TRIPLE OVERLAP parameters (PSD + Both Fitting)
        # if len(frequency_analysis_types) > 1:
        #     result = self._resolve_triple_overlap_parameters(
        #         result, frequency_analysis_types, TRIPLE_OVERLAP_PARAMS
        #     )

        if len(frequency_analysis_types) > 1:
            result = self._resolve_triple_overlap_parameters_simple(
                result, frequency_analysis_types, TRIPLE_OVERLAP_PARAMS
            )
        
        # Step 3: Handle FITTING SHARED parameters (Both Fitting Models)
        if len(fitting_types) > 1:
            result = self._resolve_fitting_shared_parameters(
                result, fitting_types, FITTING_SHARED_PARAMS
            )
        
        # Step 4: Handle SPECIAL SHARED parameters (A0 with different defaults)
        if len(fitting_types) > 1:
            result = self._resolve_special_shared_parameters(
                result, fitting_types, SPECIAL_SHARED_PARAMS
            )
        
        # Step 5: Fill defaults for each analysis type
        for analysis_type in result.analysis_types:
            if analysis_type in result.parameters:
                params = result.parameters[analysis_type]
                schema = self.parameter_schemas.get(analysis_type, {})
                defaults = schema.get("defaults", {})
                
                # Fill missing parameters with defaults
                for param_name, default_value in defaults.items():
                    if param_name not in params:
                        params[param_name] = default_value
                        
                        # Update parameter source
                        if analysis_type not in result.parameter_source:
                            result.parameter_source[analysis_type] = "defaults_used"
                        elif result.parameter_source[analysis_type] == "user_specified":
                            result.parameter_source[analysis_type] = "mixed"
                
                # Validate parameter ranges
                validation_errors = self._validate_parameter_ranges(analysis_type, params)
                if validation_errors:
                    result.potential_issues.extend(validation_errors)
            else:
                # Use defaults for unspecified analysis types
                schema = self.parameter_schemas.get(analysis_type, {})
                defaults = schema.get("defaults", {})
                if defaults:
                    result.parameters[analysis_type] = defaults.copy()
                    result.parameter_source[analysis_type] = "defaults_used"
                    result.parameter_confidence[analysis_type] = 0.8
        
        # Step 6: Final consistency validation
        consistency_errors = self._validate_comprehensive_parameter_consistency(
            result.parameters, frequency_analysis_types, fitting_types
        )
        if consistency_errors:
            result.potential_issues.extend(consistency_errors)
        
        # Step 7: Remove extra parameters
        requested_types = set(result.analysis_types)
        current_param_types = set(result.parameters.keys())
        extra_types = current_param_types - requested_types
        
        for extra_type in extra_types:
            result.parameters.pop(extra_type, None)
            result.parameter_source.pop(extra_type, None)
            result.parameter_confidence.pop(extra_type, None)
        
        return result

    # def _resolve_triple_overlap_parameters(self, 
    #                                     result: UnifiedFITSResult, 
    #                                     frequency_analysis_types: List[str],
    #                                     triple_params: set) -> UnifiedFITSResult:
    #     """Resolve parameters that overlap between PSD + Both Fitting Models"""
        
    #     for param_name in triple_params:
    #         # Collect values from all frequency analysis types
    #         param_values = {}
    #         for analysis_type in frequency_analysis_types:
    #             if (analysis_type in result.parameters and 
    #                 param_name in result.parameters[analysis_type]):
    #                 param_values[analysis_type] = result.parameters[analysis_type][param_name]
            
    #         if len(param_values) == 0:
    #             continue
    #         elif len(set(param_values.values())) == 1:
    #             # All values are the same, propagate to all types
    #             shared_value = list(param_values.values())[0]
    #             for analysis_type in frequency_analysis_types:
    #                 if analysis_type not in result.parameters:
    #                     result.parameters[analysis_type] = {}
    #                 result.parameters[analysis_type][param_name] = shared_value
    #         else:
    #             # Conflict detected! Use resolution strategy
    #             conflict_values = param_values.copy()
    #             last_mentioned_value = list(param_values.values())[-1]
                
    #             # Apply resolved value to all frequency analysis types
    #             for analysis_type in frequency_analysis_types:
    #                 if analysis_type not in result.parameters:
    #                     result.parameters[analysis_type] = {}
    #                 result.parameters[analysis_type][param_name] = last_mentioned_value
                
    #             # Add explanation for conflict resolution
    #             explanation = f"Resolved {param_name} conflict: used last mentioned value ({last_mentioned_value}) across {len(frequency_analysis_types)} analysis types"
    #             result.parameter_explanations[f"{param_name}_conflict_resolution"] = explanation
                
    #             # Update parameter source to indicate shared resolution
    #             for analysis_type in frequency_analysis_types:
    #                 if analysis_type in result.parameter_source:
    #                     if result.parameter_source[analysis_type] != "user_specified":
    #                         result.parameter_source[analysis_type] = "shared_resolution"
    #                 else:
    #                     result.parameter_source[analysis_type] = "shared_resolution"
        
    #     return result

    def _resolve_triple_overlap_parameters_simple(self, 
                                                result: UnifiedFITSResult, 
                                                frequency_analysis_types: List[str],
                                                triple_params: set) -> UnifiedFITSResult:
        """
        SIMPLER APPROACH: Only resolve when there's actual ambiguity
        If each analysis type has different explicit values, keep them as-is
        """
        
        for param_name in triple_params:
            # Collect values from all frequency analysis types
            param_values = {}
            for analysis_type in frequency_analysis_types:
                if (analysis_type in result.parameters and 
                    param_name in result.parameters[analysis_type]):
                    param_values[analysis_type] = result.parameters[analysis_type][param_name]
            
            if len(param_values) == 0:
                continue
            elif len(set(param_values.values())) == 1:
                # All values are the same, propagate to all types that don't have it
                shared_value = list(param_values.values())[0]
                for analysis_type in frequency_analysis_types:
                    if analysis_type not in result.parameters:
                        result.parameters[analysis_type] = {}
                    if param_name not in result.parameters[analysis_type]:
                        result.parameters[analysis_type][param_name] = shared_value
            else:
                # Different values exist
                # NEW LOGIC: Only resolve if we have partial specification
                
                total_types = len(frequency_analysis_types)
                specified_types = len(param_values)
                
                if specified_types == total_types:
                    # Each analysis type has its own value - keep as explicit specification
                    explanation = f"Using explicit {param_name} values per analysis type: {param_values}"
                    result.parameter_explanations[f"{param_name}_explicit_values"] = explanation
                    
                    # No changes needed - values are already correctly assigned
                    
                else:
                    # Partial specification - apply conflict resolution
                    last_mentioned_value = list(param_values.values())[-1]
                    
                    # Apply to all types
                    for analysis_type in frequency_analysis_types:
                        if analysis_type not in result.parameters:
                            result.parameters[analysis_type] = {}
                        result.parameters[analysis_type][param_name] = last_mentioned_value
                    
                    explanation = f"Resolved {param_name} conflict: used last mentioned value ({last_mentioned_value}) across {total_types} analysis types"
                    result.parameter_explanations[f"{param_name}_conflict_resolution"] = explanation
        
        return result

    def _resolve_fitting_shared_parameters(self, 
                                        result: UnifiedFITSResult,
                                        fitting_types: List[str], 
                                        fitting_params: set) -> UnifiedFITSResult:
        """Resolve parameters shared between fitting models only"""
        
        for param_name in fitting_params:
            # Collect values from fitting types
            param_values = {}
            for fitting_type in fitting_types:
                if (fitting_type in result.parameters and 
                    param_name in result.parameters[fitting_type]):
                    param_values[fitting_type] = result.parameters[fitting_type][param_name]
            
            if len(param_values) == 0:
                continue
            elif len(set(param_values.values())) == 1:
                # All values are the same, propagate to all fitting types
                shared_value = list(param_values.values())[0]
                for fitting_type in fitting_types:
                    if fitting_type not in result.parameters:
                        result.parameters[fitting_type] = {}
                    result.parameters[fitting_type][param_name] = shared_value
            else:
                # Conflict detected
                resolved_value = list(param_values.values())[-1]  # Last mentioned
                
                # Apply resolved value to all fitting types
                for fitting_type in fitting_types:
                    if fitting_type not in result.parameters:
                        result.parameters[fitting_type] = {}
                    result.parameters[fitting_type][param_name] = resolved_value
                
                # Add explanation
                explanation = f"Resolved {param_name} across {len(fitting_types)} fitting models: used last mentioned value ({resolved_value})"
                result.parameter_explanations[f"{param_name}_fitting_shared"] = explanation
        
        return result

    def _resolve_special_shared_parameters(self, 
                                        result: UnifiedFITSResult,
                                        fitting_types: List[str], 
                                        special_params: set) -> UnifiedFITSResult:
        """Handle A0 parameter with special logic for different defaults"""
        
        if "A0" not in special_params or len(fitting_types) <= 1:
            return result
        
        # Check A0 specifications
        a0_values = {}
        for fitting_type in fitting_types:
            if (fitting_type in result.parameters and 
                "A0" in result.parameters[fitting_type]):
                a0_values[fitting_type] = result.parameters[fitting_type]["A0"]
        
        if len(a0_values) == 0:
            # No A0 specified - use model-specific defaults
            for fitting_type in fitting_types:
                if fitting_type not in result.parameters:
                    result.parameters[fitting_type] = {}
                
                if fitting_type == "fitting_power_law":
                    result.parameters[fitting_type]["A0"] = 1.0
                elif fitting_type == "fitting_bending_power_law":
                    result.parameters[fitting_type]["A0"] = 10.0
            
            explanation = "Used model-specific A0 defaults: power_law=1.0, bending=10.0 (optimized for each model)"
            result.parameter_explanations["A0_model_defaults"] = explanation
        
        elif len(a0_values) == 1:
            # One A0 specified - could be intentional for one model or general
            specified_type = list(a0_values.keys())[0]
            specified_value = a0_values[specified_type]
            
            # Apply to the specified model, use defaults for others
            for fitting_type in fitting_types:
                if fitting_type not in result.parameters:
                    result.parameters[fitting_type] = {}
                
                if fitting_type == specified_type:
                    result.parameters[fitting_type]["A0"] = specified_value
                else:
                    # Use model-specific default for unspecified
                    if fitting_type == "fitting_power_law":
                        result.parameters[fitting_type]["A0"] = 1.0
                    elif fitting_type == "fitting_bending_power_law":
                        result.parameters[fitting_type]["A0"] = 10.0
            
            explanation = f"A0={specified_value} applied to {specified_type}, others use model-specific defaults"
            result.parameter_explanations["A0_partial_specification"] = explanation
        
        else:
            # Multiple A0 specified - use as specified
            for fitting_type in fitting_types:
                if fitting_type not in result.parameters:
                    result.parameters[fitting_type] = {}
                
                if fitting_type in a0_values:
                    result.parameters[fitting_type]["A0"] = a0_values[fitting_type]
                else:
                    # Use default for unspecified
                    if fitting_type == "fitting_power_law":
                        result.parameters[fitting_type]["A0"] = 1.0
                    elif fitting_type == "fitting_bending_power_law":
                        result.parameters[fitting_type]["A0"] = 10.0
            
            explanation = f"Model-specific A0 values applied: {a0_values}"
            result.parameter_explanations["A0_model_specific"] = explanation
        
        return result

    def _validate_comprehensive_parameter_consistency(self, 
                                                    parameters: Dict[str, Dict[str, Any]],
                                                    frequency_analysis_types: List[str],
                                                    fitting_types: List[str]) -> List[str]:
        """Validate consistency across all shared parameter groups"""
        
        errors = []
        
        # Check triple overlap consistency (PSD + Fitting)
        triple_params = ["low_freq", "high_freq", "bins"]
        if len(frequency_analysis_types) > 1:
            for param_name in triple_params:
                values = []
                types_with_param = []
                
                for analysis_type in frequency_analysis_types:
                    if analysis_type in parameters and param_name in parameters[analysis_type]:
                        values.append(parameters[analysis_type][param_name])
                        types_with_param.append(analysis_type)
                
                if len(set(values)) > 1:
                    errors.append(f"Inconsistent triple overlap {param_name}: {dict(zip(types_with_param, values))}")
        
        # Check fitting-only shared consistency
        fitting_params = ["noise_bound_percent", "A_min", "A_max", "maxfev"]
        if len(fitting_types) > 1:
            for param_name in fitting_params:
                values = []
                types_with_param = []
                
                for fitting_type in fitting_types:
                    if fitting_type in parameters and param_name in parameters[fitting_type]:
                        values.append(parameters[fitting_type][param_name])
                        types_with_param.append(fitting_type)
                
                if len(set(values)) > 1:
                    errors.append(f"Inconsistent fitting shared {param_name}: {dict(zip(types_with_param, values))}")
        
        # Note: A0 is intentionally NOT validated for consistency since it can have different values
        
        return errors
    
    def _validate_parameter_ranges(self, analysis_type: str, params: Dict[str, Any]) -> List[str]:
        """Validate parameter ranges for analysis type"""
        errors = []
        schema = self.parameter_schemas.get(analysis_type, {})
        param_defs = schema.get("parameters", {})
        
        for param_name, param_value in params.items():
            if param_name in param_defs:
                param_def = param_defs[param_name]
                param_range = param_def.get("range")
                
                if param_range and isinstance(param_value, (int, float)):
                    min_val, max_val = param_range
                    if param_value < min_val or param_value > max_val:
                        errors.append(f"{analysis_type}.{param_name} ({param_value}) outside valid range [{min_val}, {max_val}]")
        
        # Cross-parameter validation
        if analysis_type in ["psd", "fitting_power_law", "fitting_bending_power_law"]:
            if "low_freq" in params and "high_freq" in params:
                if params["low_freq"] >= params["high_freq"]:
                    errors.append(f"{analysis_type}: low_freq must be < high_freq")
        
        return errors
    
    def _create_fallback_result(self, user_input: str, error: str) -> UnifiedFITSResult:
        """Create fallback result when processing fails - ORCHESTRATOR COMPATIBLE"""
        return UnifiedFITSResult(
            primary_intent="analysis",
            analysis_types=["statistics"],  # Safe default
            routing_strategy="analysis",
            confidence=0.3,
            reasoning=f"Fallback result due to error: {error}",
            question_category="unknown",  # ✅ ORCHESTRATOR REQUIRED
            complexity_level="intermediate",  # ✅ ORCHESTRATOR REQUIRED
            parameters={"statistics": self.parameter_schemas["statistics"]["defaults"]},
            parameter_confidence={"statistics": 0.3},
            parameter_source={"statistics": "defaults_used"},
            suggested_workflow=["check_input", "retry_request"],
            potential_issues=[f"processing_error: {error}"],
            requires_file=True
        )
    
    def _create_fallback_parsing_result(self, raw_output: str) -> UnifiedFITSResult:
        """Create result when JSON parsing fails but we have text - ORCHESTRATOR COMPATIBLE"""
        
        # Try to extract basic intent from text
        text_lower = raw_output.lower()
        
        if "mixed" in text_lower:
            primary_intent = "mixed"
            routing_strategy = "mixed"
            astrosage_required = True
        elif "general" in text_lower or "question" in text_lower:
            primary_intent = "general"
            routing_strategy = "astrosage"
            astrosage_required = True
        else:
            primary_intent = "analysis"
            routing_strategy = "analysis"
            astrosage_required = False
        
        # Extract analysis types mentioned
        analysis_types = []
        if "statistics" in text_lower or "mean" in text_lower:
            analysis_types.append("statistics")
        if "psd" in text_lower or "power spectral" in text_lower:
            analysis_types.append("psd")
        if "fitting" in text_lower or "power law" in text_lower:
            if "bending" in text_lower:
                analysis_types.append("fitting_bending_power_law")
            else:
                analysis_types.append("fitting_power_law")
        
        if not analysis_types and primary_intent == "analysis":
            analysis_types = ["statistics"]  # Safe default
        
        # Determine question category from text
        question_category = "unknown"
        if "neutron star" in text_lower or "black hole" in text_lower or "stellar" in text_lower:
            question_category = "astronomy"
        elif "psd" in text_lower or "statistics" in text_lower or "fitting" in text_lower:
            question_category = "data_analysis"
        elif "physics" in text_lower or "quantum" in text_lower or "emission" in text_lower:
            question_category = "physics"
        elif "how" in text_lower and ("analyze" in text_lower or "method" in text_lower):
            question_category = "methods"
        
        # Build parameters with defaults
        parameters = {}
        parameter_confidence = {}
        parameter_source = {}
        for analysis_type in analysis_types:
            if analysis_type in self.parameter_schemas:
                parameters[analysis_type] = self.parameter_schemas[analysis_type]["defaults"]
                parameter_confidence[analysis_type] = 0.5
                parameter_source[analysis_type] = "defaults_used"
        
        return UnifiedFITSResult(
            primary_intent=primary_intent,
            analysis_types=analysis_types,
            routing_strategy=routing_strategy,
            confidence=0.6,
            reasoning="Fallback text analysis due to JSON parsing failure",
            question_category=question_category,  # ✅ ORCHESTRATOR REQUIRED
            complexity_level="intermediate",  # ✅ ORCHESTRATOR REQUIRED
            parameters=parameters,
            parameter_confidence=parameter_confidence,
            parameter_source=parameter_source,
            astrosage_required=astrosage_required,
            suggested_workflow=["verify_request", "reprocess_if_needed"],
            potential_issues=["json_parsing_failed"],
            requires_file=(primary_intent == "analysis")
        )
    
    def _cleanup_cache(self):
        """Clean up expired cache entries"""
        current_time = datetime.now()
        expired_keys = []
        
        for cache_key, timestamp in self.cache_timestamps.items():
            if current_time - timestamp > timedelta(seconds=self.cache_ttl):
                expired_keys.append(cache_key)
        
        for key in expired_keys:
            self.cache.pop(key, None)
            self.cache_timestamps.pop(key, None)
    
    def _generate_cache_key(self, user_input: str, context: Dict[str, Any]) -> str:
        """Generate cache key from input and context"""
        normalized_input = re.sub(r'\s+', ' ', user_input.lower().strip())
        context_key = {
            "has_files": context.get("has_uploaded_files", False),
            "expertise": context.get("user_expertise", "intermediate"),
            "prev_analyses": sorted(context.get("previous_analyses", []))
        }
        
        combined = f"{normalized_input}:{json.dumps(context_key, sort_keys=True)}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics with question categorization"""
        total = max(self.stats["total_requests"], 1)
        
        return {
            "usage": {
                "total_requests": self.stats["total_requests"],
                "cache_hits": self.stats["cache_hits"],
                "cache_hit_rate": round(self.stats["cache_hits"] / total, 3),
                "total_tokens_used": self.stats["total_tokens"],
                "total_cost": round(self.stats["total_cost"], 4),
                "avg_tokens_per_request": round(self.stats["total_tokens"] / total, 1),
                "avg_cost_per_request": round(self.stats["total_cost"] / total, 6),
                "avg_processing_time": round(self.stats["avg_processing_time"], 3)
            },
            "classification_distribution": {
                "analysis_requests": self.stats["intent_distribution"]["analysis"],
                "general_questions": self.stats["intent_distribution"]["general"],
                "mixed_requests": self.stats["intent_distribution"]["mixed"],
                "analysis_percentage": round(self.stats["intent_distribution"]["analysis"] / total * 100, 1),
                "general_percentage": round(self.stats["intent_distribution"]["general"] / total * 100, 1),
                "mixed_percentage": round(self.stats["intent_distribution"]["mixed"] / total * 100, 1)
            },
            "question_categories": {
                "astronomy": self.stats["question_categories"]["astronomy"],
                "physics": self.stats["question_categories"]["physics"],
                "data_analysis": self.stats["question_categories"]["data_analysis"],
                "methods": self.stats["question_categories"]["methods"],
                "unknown": self.stats["question_categories"]["unknown"],
                "astronomy_percentage": round(self.stats["question_categories"]["astronomy"] / total * 100, 1),
                "physics_percentage": round(self.stats["question_categories"]["physics"] / total * 100, 1),
                "data_analysis_percentage": round(self.stats["question_categories"]["data_analysis"] / total * 100, 1),
                "methods_percentage": round(self.stats["question_categories"]["methods"] / total * 100, 1)
            },
            "parameter_stats": {
                "parameter_extractions": self.stats["parameter_extractions"],
                "parameter_extraction_rate": round(self.stats["parameter_extractions"] / total, 3)
            },
            "cost_projections": {
                "cost_per_1000_requests": round(self.stats["total_cost"] / total * 1000, 2),
                "monthly_cost_10k_requests": round(self.stats["total_cost"] / total * 10000, 2),
                "monthly_cost_100k_requests": round(self.stats["total_cost"] / total * 100000, 2)
            },
            "cache_info": {
                "cache_size": len(self.cache),
                "cache_efficiency": round(self.stats["cache_hits"] / total, 3) if total > 0 else 0
            }
        }
    
    def clear_cache(self):
        """Clear the cache"""
        self.cache.clear()
        self.cache_timestamps.clear()
        self.logger.info("Cache cleared")
    
    def get_parameter_schema(self, analysis_type: str) -> Dict[str, Any]:
        """Get parameter schema for specific analysis type"""
        return self.parameter_schemas.get(analysis_type, {})
    
    def validate_parameters(self, analysis_type: str, parameters: Dict[str, Any]) -> List[str]:
        """Validate parameters for specific analysis type"""
        return self._validate_parameter_ranges(analysis_type, parameters)


# ========================================
# COMPREHENSIVE TEST SUITE - ORCHESTRATOR COMPATIBILITY
# ========================================

async def test_orchestrator_compatibility():
    """Test orchestrator compatibility specifically"""
    print("🎯 Testing Orchestrator Compatibility")
    print("=" * 60)
    
    agent = UnifiedFITSClassificationAgent()
    
    # Test cases that previously failed
    compatibility_tests = [
        {
            "name": "General Question - Orchestrator Error Test",
            "input": "What is a power spectral density?",
            "context": {"user_expertise": "beginner"},
            "expected_attributes": [
                "question_category", "complexity_level", "question_context",
                "primary_intent", "analysis_types", "confidence"
            ]
        },
        {
            "name": "Mixed Request - Question then Analysis",
            "input": "What is PSD? Then compute it for my data with 4000 bins",
            "context": {"has_uploaded_files": True, "user_expertise": "intermediate"},
            "expected_attributes": [
                "question_category", "complexity_level", "question_context",
                "is_mixed_request", "primary_intent", "analysis_types"
            ]
        },
        {
            "name": "Complex Mixed Request",
            "input": "Compute PSD and fit both power law models, then explain what the break frequency means",
            "context": {"has_uploaded_files": True, "user_expertise": "advanced"},
            "expected_attributes": [
                "question_category", "complexity_level", "question_context",
                "is_mixed_request", "parameters", "parameter_source"
            ]
        }
    ]
    
    success_count = 0
    
    for i, test in enumerate(compatibility_tests, 1):
        print(f"\nTest {i}: {test['name']}")
        print(f"Input: '{test['input']}'")
        
        try:
            result = await agent.process_request(test["input"], test["context"])
            
            print(f"✅ Processing successful")
            print(f"   Primary Intent: {result.primary_intent}")
            print(f"   Question Category: {result.question_category}")
            print(f"   Complexity Level: {result.complexity_level}")
            print(f"   Analysis Types: {result.analysis_types}")
            print(f"   Mixed Request: {result.is_mixed_request}")
            if result.question_context:
                print(f"   Question Context: {result.question_context}")
            
            # Check all required attributes exist
            missing_attrs = []
            for attr in test["expected_attributes"]:
                if not hasattr(result, attr):
                    missing_attrs.append(attr)
                elif getattr(result, attr) is None and attr in ["question_category", "complexity_level"]:
                    missing_attrs.append(f"{attr} is None")
            
            if missing_attrs:
                print(f"   ❌ Missing attributes: {missing_attrs}")
            else:
                print(f"   ✅ All required attributes present")
                success_count += 1
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    
    print(f"\n{'='*60}")
    print(f"ORCHESTRATOR COMPATIBILITY RESULTS")
    print(f"{'='*60}")
    print(f"Success Rate: {success_count}/{len(compatibility_tests)}")
    
    if success_count == len(compatibility_tests):
        print("🎉 ALL COMPATIBILITY TESTS PASSED!")
        print("✅ Agent is fully compatible with FITSAnalysisOrchestrator")
    else:
        print("❌ Some compatibility issues remain")
    
    return success_count == len(compatibility_tests)


async def test_unified_fits_agent_with_enhanced_cases():
    """Enhanced test suite combining both versions - Compatible with Version 2.1"""
    print("🧪 Testing Unified FITS Classification + Parameter Agent v2.1")
    print("=" * 80)
    
    agent = UnifiedFITSClassificationAgent()
    
    # ✅ ENHANCED TEST CASES - รวมจาก paste-2.txt + Version 2.1 features
    test_cases = [
        {
            "name": "Pure Statistics Analysis",
            "input": "Calculate mean, median, and standard deviation for my lightcurve data",
            "context": {"has_uploaded_files": True, "user_expertise": "intermediate"},
            "expected": {
                "primary_intent": "analysis",
                "analysis_types": ["statistics"],
                "routing_strategy": "analysis",
                "question_category": "unknown",  #  เพิ่มใหม่ใน v2.1
                "complexity_level": "intermediate",  #  เพิ่มใหม่ใน v2.1
                "has_parameters": True,
                "parameter_types": ["statistics"],
                "astrosage_required": False  #  เพิ่มเติม
            }
        },
        {
            "name": "PSD Analysis with Custom Parameters",
            "input": "Compute power spectral density with frequency range 1e-4 to 0.1 Hz and 5000 bins",
            "context": {"has_uploaded_files": True, "user_expertise": "advanced"},
            "expected": {
                "primary_intent": "analysis",
                "analysis_types": ["psd"],
                "routing_strategy": "analysis",
                "question_category": "unknown",  #  ไม่มีคำถาม
                "complexity_level": "advanced",  #  จาก context
                "has_parameters": True,
                "parameter_types": ["psd"],
                "custom_params": {"low_freq": 1e-4, "high_freq": 0.1, "bins": 5000},
                "astrosage_required": False
            }
        },
        {
            "name": "Power Law Fitting with Initial Guesses",
            "input": "Fit a power law model with initial amplitude of 2.5 and power index of 1.8",
            "context": {"has_uploaded_files": True, "user_expertise": "advanced"},
            "expected": {
                "primary_intent": "analysis",
                "analysis_types": ["fitting_power_law"],
                "routing_strategy": "analysis",
                "question_category": "unknown",
                "complexity_level": "advanced",
                "has_parameters": True,
                "parameter_types": ["fitting_power_law"],
                "custom_params": {"A0": 2.5, "b0": 1.8},
                "astrosage_required": False
            }
        },
        {
            "name": "Pure General Question - Astronomy",
            "input": "What causes neutron stars to emit X-rays?",
            "context": {"user_expertise": "beginner"},
            "expected": {
                "primary_intent": "general",
                "analysis_types": [],
                "routing_strategy": "astrosage",
                "question_category": "astronomy",  #  เพิ่มใหม่ - คำถามดาราศาสตร์
                "complexity_level": "beginner",
                "has_parameters": False,
                "astrosage_required": True  #  สำคัญสำหรับ v2.1
            }
        },
        {
            "name": "Data Analysis Question",
            "input": "What is power spectral density and how do I interpret the results?",
            "context": {"user_expertise": "intermediate"},
            "expected": {
                "primary_intent": "general",
                "analysis_types": [],
                "routing_strategy": "astrosage",
                "question_category": "data_analysis",  #  เพิ่มใหม่ - คำถามการวิเคราะห์ข้อมูล
                "complexity_level": "intermediate",
                "has_parameters": False,
                "astrosage_required": True
            }
        },
        {
            "name": "Mixed Request - Question then Analysis",
            "input": "What is a bending power law? Then fit one to my data with break frequency around 0.01 Hz",
            "context": {"has_uploaded_files": True, "user_expertise": "intermediate"},
            "expected": {
                "primary_intent": "mixed",  #  สำคัญ - v2.1 ตรวจจับ mixed ได้ดีกว่า
                "analysis_types": ["fitting_bending_power_law"],
                "routing_strategy": "mixed",
                "question_category": "data_analysis",  #  เพิ่มใหม่ - เกี่ยวกับ bending power law
                "complexity_level": "intermediate",
                "has_parameters": True,
                "parameter_types": ["fitting_bending_power_law"],
                "custom_params": {"fb0": 0.01},
                "is_mixed_request": True,  #  เพิ่มเติม
                "astrosage_required": True  #  ต้องการ AstroSage เพื่ออธิบาย
            }
        },
        {
            "name": "Complex Sequential Analysis",
            "input": "Perform complete timing analysis: compute PSD with 4000 bins, then fit both power law and bending power law models",
            "context": {"has_uploaded_files": True, "user_expertise": "expert"},
            "expected": {
                "primary_intent": "analysis",
                "analysis_types": ["psd", "fitting_power_law", "fitting_bending_power_law"],
                "routing_strategy": "analysis",
                "question_category": "unknown",  #  ไม่มีคำถาม แค่ analysis
                "complexity_level": "expert",  #  จาก context -> advanced
                "has_parameters": True,
                "parameter_types": ["psd", "fitting_power_law", "fitting_bending_power_law"],
                "custom_params": {"bins": 4000},
                "astrosage_required": False
            }
        },
        {
            "name": "Beginner Mixed Request - Enhanced Detection",
            "input": "I'm new to X-ray astronomy. Calculate basic statistics for my neutron star data and explain what they mean",
            "context": {"has_uploaded_files": True, "user_expertise": "beginner"},
            "expected": {
                "primary_intent": "mixed",  #  v2.1 จับได้ดีกว่า
                "analysis_types": ["statistics"],
                "routing_strategy": "mixed",
                "question_category": "astronomy",  #  เพิ่มใหม่ - บริบท neutron star
                "complexity_level": "beginner",
                "has_parameters": True,
                "parameter_types": ["statistics"],
                "is_mixed_request": True,
                "astrosage_required": True  #  ต้องการอธิบาย
            }
        },
        {
            "name": "Physics Question",
            "input": "Explain the physics behind X-ray emission from accreting black holes",
            "context": {"user_expertise": "advanced"},
            "expected": {
                "primary_intent": "general",
                "analysis_types": [],
                "routing_strategy": "astrosage",
                "question_category": "physics",  #  เพิ่มใหม่ - คำถามฟิสิกส์
                "complexity_level": "advanced",
                "has_parameters": False,
                "astrosage_required": True
            }
        },
        {
            "name": "Methods Question",
            "input": "What's the best way to fit power law models to noisy data?",
            "context": {"user_expertise": "intermediate"},
            "expected": {
                "primary_intent": "general",
                "analysis_types": [],
                "routing_strategy": "astrosage",
                "question_category": "methods",  #  เพิ่มใหม่ - คำถามเกี่ยวกับวิธีการ
                "complexity_level": "intermediate",
                "has_parameters": False,
                "astrosage_required": True
            }
        },
        {
            "name": "Advanced Mixed Request - Analysis + Interpretation",
            "input": "Calculate statistics and compute PSD, then explain what the frequency peaks might indicate about the neutron star's rotation",
            "context": {"has_uploaded_files": True, "user_expertise": "intermediate"},
            "expected": {
                "primary_intent": "mixed",
                "analysis_types": ["statistics", "psd"],
                "routing_strategy": "mixed",
                "question_category": "astronomy",  #  เพิ่มใหม่ - บริบท neutron star rotation
                "complexity_level": "intermediate",
                "has_parameters": True,
                "parameter_types": ["statistics", "psd"],
                "is_mixed_request": True,
                "astrosage_required": True
            }
        },
        {
            "name": "Bending Power Law with Custom Parameters - Enhanced",
            "input": "Fit bending power law with shape parameter 2.0 and amplitude bounds from 1 to 100",
            "context": {"has_uploaded_files": True, "user_expertise": "advanced"},
            "expected": {
                "primary_intent": "analysis",
                "analysis_types": ["fitting_bending_power_law"],
                "routing_strategy": "analysis",
                "question_category": "unknown",
                "complexity_level": "advanced",
                "has_parameters": True,
                "parameter_types": ["fitting_bending_power_law"],
                "custom_params": {"sh0": 2.0, "A_min": 1.0, "A_max": 100.0},
                "astrosage_required": False
            }
        }
    ]
    
    success_count = 0
    total_cost = 0.0
    total_time = 0.0
    detailed_results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*90}")
        print(f"Test {i}: {test_case['name']}")
        print(f"Input: '{test_case['input']}'")
        print(f"Context: {test_case['context']}")
        print(f"{'='*90}")
        
        try:
            start_time = datetime.now()
            result = await agent.process_request(test_case["input"], test_case["context"])
            processing_time = (datetime.now() - start_time).total_seconds()
            
            total_cost += result.cost_estimate
            total_time += processing_time
            
            # Display results
            print(f"✅ Processing successful in {processing_time:.3f}s")
            print(f"💰 Cost: ${result.cost_estimate:.6f}")
            print(f"🎯 Confidence: {result.confidence:.2f}")
            
            print(f"\n📊 Classification Results:")
            print(f"   Primary Intent: {result.primary_intent}")
            print(f"   Analysis Types: {result.analysis_types}")
            print(f"   Routing Strategy: {result.routing_strategy}")
            print(f"   Question Category: {result.question_category}")  #  เพิ่มใหม่
            print(f"   Complexity Level: {result.complexity_level}")      #  เพิ่มใหม่
            print(f"   Mixed Request: {result.is_mixed_request}")
            print(f"   AstroSage Required: {result.astrosage_required}")  #  เพิ่มใหม่
            
            if result.parameters:
                print(f"\n⚙️  Parameters:")
                for analysis_type, params in result.parameters.items():
                    print(f"   {analysis_type}: {params}")
            
            # ✅ ENHANCED VALIDATION for Version 2.1
            expected = test_case["expected"]
            validations = []
            
            # Core validations (เหมือนเดิม)
            intent_ok = result.primary_intent == expected["primary_intent"]
            validations.append(("Primary Intent", intent_ok, expected["primary_intent"], result.primary_intent))
            
            analysis_ok = set(result.analysis_types) == set(expected["analysis_types"])
            validations.append(("Analysis Types", analysis_ok, expected["analysis_types"], result.analysis_types))
            
            routing_ok = result.routing_strategy == expected["routing_strategy"]
            validations.append(("Routing Strategy", routing_ok, expected["routing_strategy"], result.routing_strategy))
            
            #  NEW VALIDATIONS for Version 2.1
            category_ok = result.question_category == expected["question_category"]
            validations.append(("Question Category", category_ok, expected["question_category"], result.question_category))
            
            complexity_ok = result.complexity_level == expected["complexity_level"]
            validations.append(("Complexity Level", complexity_ok, expected["complexity_level"], result.complexity_level))
            
            # Mixed request validation (ถ้ามี)
            if "is_mixed_request" in expected:
                mixed_ok = result.is_mixed_request == expected["is_mixed_request"]
                validations.append(("Mixed Request", mixed_ok, expected["is_mixed_request"], result.is_mixed_request))
            
            # AstroSage requirement validation
            if "astrosage_required" in expected:
                astrosage_ok = result.astrosage_required == expected["astrosage_required"]
                validations.append(("AstroSage Required", astrosage_ok, expected["astrosage_required"], result.astrosage_required))
            
            # Parameter validations (เหมือนเดิม)
            has_params_ok = bool(result.parameters) == expected["has_parameters"]
            validations.append(("Has Parameters", has_params_ok, expected["has_parameters"], bool(result.parameters)))
            
            if expected.get("parameter_types"):
                param_types_ok = set(result.parameters.keys()) == set(expected["parameter_types"])
                validations.append(("Parameter Types", param_types_ok, expected["parameter_types"], list(result.parameters.keys())))
            
            # Custom parameter validation (เหมือนเดิม)
            if expected.get("custom_params"):
                custom_params_ok = True
                custom_params_found = {}
                
                for param_key, expected_value in expected["custom_params"].items():
                    found = False
                    actual_value = None
                    
                    for analysis_type, analysis_params in result.parameters.items():
                        if param_key in analysis_params:
                            actual_value = analysis_params[param_key]
                            
                            if isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
                                if abs(actual_value - expected_value) < 1e-6:
                                    found = True
                                    custom_params_found[param_key] = actual_value
                                    break
                            elif actual_value == expected_value:
                                found = True
                                custom_params_found[param_key] = actual_value
                                break
                    
                    if not found:
                        custom_params_ok = False
                        break
                
                validations.append(("Custom Parameters", custom_params_ok, expected["custom_params"], custom_params_found))
            
            #  VERSION 2.1 ORCHESTRATOR COMPATIBILITY CHECK
            orchestrator_attrs = ["question_category", "complexity_level", "primary_intent", "analysis_types", "confidence"]
            missing_attrs = []
            for attr in orchestrator_attrs:
                if not hasattr(result, attr):
                    missing_attrs.append(attr)
                elif getattr(result, attr) is None and attr in ["question_category", "complexity_level"]:
                    missing_attrs.append(f"{attr}_is_None")
            
            orchestrator_ok = len(missing_attrs) == 0
            validations.append(("Orchestrator Compatibility", orchestrator_ok, "All required attributes", missing_attrs or "All present"))
            
            print(f"\n✅ Validation Results:")
            all_valid = True
            for check_name, is_valid, expected_val, actual_val in validations:
                status = "✅" if is_valid else "❌"
                print(f"   {check_name}: {status}")
                if not is_valid:
                    print(f"      Expected: {expected_val}")
                    print(f"      Got: {actual_val}")
                    all_valid = False
            
            if all_valid:
                success_count += 1
                print(f"\n🎉 PASS")
            else:
                print(f"\n❌ FAIL")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        await asyncio.sleep(0.3)

        print("Results :", result)
    
    # Final summary
    print(f"\n{'='*90}")
    print("🎯 ENHANCED UNIFIED FITS AGENT v2.1 TEST RESULTS")
    print(f"{'='*90}")
    
    print(f"✅ Success Rate: {success_count}/{len(test_cases)} ({success_count/len(test_cases)*100:.1f}%)")
    print(f"💰 Total Cost: ${total_cost:.6f}")
    print(f"⏱️  Total Time: {total_time:.3f}s")
    print(f"📊 Avg Cost per Request: ${total_cost/len(test_cases):.6f}")
    print(f"📈 Avg Time per Request: {total_time/len(test_cases):.3f}s")
    
    #  VERSION 2.1 SPECIFIC STATISTICS
    stats = agent.get_comprehensive_stats()
    print(f"\n📊 VERSION 2.1 STATISTICS:")
    print(f"   Intent Distribution: Analysis {stats['classification_distribution']['analysis_percentage']:.1f}%, "
          f"General {stats['classification_distribution']['general_percentage']:.1f}%, "
          f"Mixed {stats['classification_distribution']['mixed_percentage']:.1f}%")
    print(f"   Question Categories: Astronomy {stats['question_categories']['astronomy_percentage']:.1f}%, "
          f"Physics {stats['question_categories']['physics_percentage']:.1f}%, "
          f"Data Analysis {stats['question_categories']['data_analysis_percentage']:.1f}%, "
          f"Methods {stats['question_categories']['methods_percentage']:.1f}%")
    
    if success_count == len(test_cases):
        print(f"\n🎉 PERFECT SCORE! ALL TESTS PASSED!")
        print(f"✅ Version 2.1 is ready for production with full orchestrator compatibility!")
    elif success_count >= len(test_cases) * 0.9:
        print(f"\n🟢 EXCELLENT! Minor refinements may be needed.")
    else:
        print(f"\n🟡 GOOD! Some improvements recommended.")
    
    return success_count == len(test_cases)


# ========================================
# ENHANCED TEST CASES FOR SHARED PARAMETERS
# ========================================

# เพิ่ม test cases ใหม่ลงใน test_unified_fits_agent_with_enhanced_cases
def get_enhanced_test_cases_for_shared_parameters():
    """
    Test cases เพิ่มเติมสำหรับทดสอบ shared parameters
    ใช้ร่วมกับ test_unified_fits_agent_with_enhanced_cases ที่มีอยู่แล้ว
    """
    
    additional_test_cases = [
        # ========== TRIPLE OVERLAP TESTS ==========
        {
            "name": "TRIPLE OVERLAP - All Same Frequency Range",
            "input": "Compute PSD and fit both power law models with frequency range 1e-4 to 0.1 Hz and 5000 bins",
            "context": {"has_uploaded_files": True, "user_expertise": "advanced"},
            "expected": {
                "primary_intent": "analysis",
                "analysis_types": ["psd", "fitting_power_law", "fitting_bending_power_law"],
                "routing_strategy": "analysis",
                "question_category": "unknown",
                "complexity_level": "advanced",
                "has_parameters": True,
                "parameter_types": ["psd", "fitting_power_law", "fitting_bending_power_law"],
                "shared_params_validation": {
                    "low_freq": 1e-4,    # Must be same across all 3 types
                    "high_freq": 0.1,    # Must be same across all 3 types
                    "bins": 5000         # Must be same across all 3 types
                },
                "different_params_validation": {
                    "fitting_power_law": {"A0": 1.0},         # Model-specific default
                    "fitting_bending_power_law": {"A0": 10.0} # Different model default
                },
                "astrosage_required": False
            }
        },
        
        {
            "name": "TRIPLE OVERLAP - Conflict Resolution",
            "input": "Compute PSD with 4000 bins, then fit power law with 5000 bins",
            "context": {"has_uploaded_files": True, "user_expertise": "intermediate"},
            "expected": {
                "primary_intent": "analysis",
                "analysis_types": ["psd", "fitting_power_law"],
                "routing_strategy": "analysis",
                "question_category": "unknown",
                "complexity_level": "intermediate",
                "has_parameters": True,
                "parameter_types": ["psd", "fitting_power_law"],
                # "shared_params_validation": {
                #     "bins": 5000  # Should use last mentioned value (5000) for both
                # },
                "different_params_validation": {
                    "psd": {"bins": 4000},           # ค่าที่ user ระบุสำหรับ PSD
                    "fitting_power_law": {"bins": 5000}  # ค่าที่ user ระบุสำหรับ fitting
                },
                "expect_conflict_explanation": True,  # Should explain resolution
                "astrosage_required": False
            }
        },
        
        # ========== A0 SPECIAL HANDLING TESTS ==========
        {
            "name": "A0 SPECIAL - No Amplitude Specified",
            "input": "Fit both power law and bending power law models",
            "context": {"has_uploaded_files": True, "user_expertise": "intermediate"},
            "expected": {
                "primary_intent": "analysis",
                "analysis_types": ["fitting_power_law", "fitting_bending_power_law"],
                "routing_strategy": "analysis",
                "question_category": "unknown",
                "complexity_level": "intermediate",
                "has_parameters": True,
                "parameter_types": ["fitting_power_law", "fitting_bending_power_law"],
                "different_params_validation": {
                    "fitting_power_law": {"A0": 1.0},         # Model default
                    "fitting_bending_power_law": {"A0": 10.0} # Different model default
                },
                "expect_a0_explanation": True,  # Should explain different defaults
                "astrosage_required": False
            }
        },
        
        {
            "name": "A0 SPECIAL - General Amplitude Specified",
            "input": "Fit both power law and bending power law with amplitude 5.0",
            "context": {"has_uploaded_files": True, "user_expertise": "advanced"},
            "expected": {
                "primary_intent": "analysis",
                "analysis_types": ["fitting_power_law", "fitting_bending_power_law"],
                "routing_strategy": "analysis",
                "question_category": "unknown",
                "complexity_level": "advanced",
                "has_parameters": True,
                "parameter_types": ["fitting_power_law", "fitting_bending_power_law"],
                "shared_params_validation": {
                    "A0": 5.0  # Both models should get A0=5.0
                },
                "expect_a0_explanation": True,  # Should explain why both get same value
                "astrosage_required": False
            }
        },
        
        {
            "name": "A0 SPECIAL - Model-Specific Amplitudes",
            "input": "Fit power law with amplitude 2.0, then fit bending power law with amplitude 15.0",
            "context": {"has_uploaded_files": True, "user_expertise": "expert"},
            "expected": {
                "primary_intent": "analysis",
                "analysis_types": ["fitting_power_law", "fitting_bending_power_law"],
                "routing_strategy": "analysis",
                "question_category": "unknown",
                "complexity_level": "expert",
                "has_parameters": True,
                "parameter_types": ["fitting_power_law", "fitting_bending_power_law"],
                "different_params_validation": {
                    "fitting_power_law": {"A0": 2.0},
                    "fitting_bending_power_law": {"A0": 15.0}
                },
                "expect_a0_explanation": True,  # Should explain model-specific values
                "astrosage_required": False
            }
        },
        
        # ========== FITTING SHARED PARAMETERS TESTS ==========
        {
            "name": "FITTING SHARED - Noise Bound Specification",
            "input": "Fit both power law and bending power law with noise bound 0.8 and max iterations 2000000",
            "context": {"has_uploaded_files": True, "user_expertise": "advanced"},
            "expected": {
                "primary_intent": "analysis",
                "analysis_types": ["fitting_power_law", "fitting_bending_power_law"],
                "routing_strategy": "analysis",
                "question_category": "unknown",
                "complexity_level": "advanced",
                "has_parameters": True,
                "parameter_types": ["fitting_power_law", "fitting_bending_power_law"],
                "shared_params_validation": {
                    "noise_bound_percent": 0.8,
                    "maxfev": 2000000
                },
                "astrosage_required": False
            }
        },
        
        # ========== COMPLEX MIXED SCENARIOS ==========
        {
            "name": "COMPLEX MIXED - All Parameter Types",
            "input": "Compute PSD with 4000 bins, fit power law with amplitude 2.0, fit bending power law with break frequency 0.02 Hz, use frequency range 1e-4 to 0.2 Hz for all, noise bound 0.8",
            "context": {"has_uploaded_files": True, "user_expertise": "expert"},
            "expected": {
                "primary_intent": "analysis",
                "analysis_types": ["psd", "fitting_power_law", "fitting_bending_power_law"],
                "routing_strategy": "analysis",
                "question_category": "unknown",
                "complexity_level": "expert", 
                "has_parameters": True,
                "parameter_types": ["psd", "fitting_power_law", "fitting_bending_power_law"],
                "shared_params_validation": {
                    "bins": 4000,          # Triple overlap
                    "low_freq": 1e-4,      # Triple overlap
                    "high_freq": 0.2,      # Triple overlap
                    "noise_bound_percent": 0.8  # Fitting shared (not in PSD)
                },
                "different_params_validation": {
                    "fitting_power_law": {"A0": 2.0},          # User specified
                    "fitting_bending_power_law": {"fb0": 0.02, "A0": 10.0}  # fb0 user specified, A0 default
                },
                "astrosage_required": False
            }
        },
        
        # ========== MIXED REQUEST WITH SHARED PARAMETERS ==========
        {
            "name": "MIXED REQUEST - Shared Parameters + Explanation",
            "input": "Compute PSD and fit power law with 5000 bins and frequency range 1e-4 to 0.1 Hz, then explain what the results indicate about timing variability",
            "context": {"has_uploaded_files": True, "user_expertise": "intermediate"},
            "expected": {
                "primary_intent": "mixed",  # Analysis + explanation
                "analysis_types": ["psd", "fitting_power_law"],
                "routing_strategy": "mixed",
                "question_category": "astronomy",  # About timing variability
                "complexity_level": "intermediate",
                "has_parameters": True,
                "parameter_types": ["psd", "fitting_power_law"],
                "shared_params_validation": {
                    "bins": 5000,
                    "low_freq": 1e-4,
                    "high_freq": 0.1
                },
                "is_mixed_request": True,
                "astrosage_required": True
            }
        },
        
        # ========== ERROR CASES ==========
        {
            "name": "PARAMETER VALIDATION - Out of Range",
            "input": "Compute PSD with 50000 bins and frequency range 0.8 to 0.9 Hz",  # bins too high, freq too high
            "context": {"has_uploaded_files": True, "user_expertise": "beginner"},
            "expected": {
                "primary_intent": "analysis",
                "analysis_types": ["psd"],
                "routing_strategy": "analysis",
                "question_category": "unknown",
                "complexity_level": "beginner",
                "has_parameters": True,
                "parameter_types": ["psd"],
                "expect_validation_issues": True,  # Should detect out-of-range parameters
                "astrosage_required": False
            }
        }
    ]
    
    return additional_test_cases

# ========================================
# ENHANCED VALIDATION FUNCTIONS
# ========================================

def validate_shared_parameters(result, expected):
    """Enhanced validation for shared parameters across multiple analysis types"""
    
    validations = []
    
    # Validate shared parameters (must be same across specified analysis types)
    if "shared_params_validation" in expected:
        for param_name, expected_value in expected["shared_params_validation"].items():
            
            # Find which analysis types should have this parameter
            relevant_types = []
            actual_values = []
            
            for analysis_type in result.analysis_types:
                if param_name in result.parameters.get(analysis_type, {}):
                    relevant_types.append(analysis_type)
                    actual_values.append(result.parameters[analysis_type][param_name])
            
            # Check consistency
            if len(set(actual_values)) == 1 and actual_values[0] == expected_value:
                validations.append((f"Shared {param_name}", True, f"{expected_value} across {relevant_types}", f"{actual_values[0]} across {relevant_types}"))
            else:
                validations.append((f"Shared {param_name}", False, f"{expected_value} across all types", f"{dict(zip(relevant_types, actual_values))}"))
    
    # Validate different parameters (must be different across analysis types)
    if "different_params_validation" in expected:
        for analysis_type, expected_params in expected["different_params_validation"].items():
            if analysis_type in result.parameters:
                for param_name, expected_value in expected_params.items():
                    actual_value = result.parameters[analysis_type].get(param_name)
                    param_ok = actual_value == expected_value
                    validations.append((f"{analysis_type}.{param_name}", param_ok, expected_value, actual_value))
    
    # Check for expected explanations
    if expected.get("expect_conflict_explanation"):
        has_conflict_explanation = any("conflict" in key.lower() for key in result.parameter_explanations.keys())
        validations.append(("Conflict Explanation", has_conflict_explanation, "Should have conflict explanation", f"Found: {has_conflict_explanation}"))
    
    if expected.get("expect_a0_explanation"):
        has_a0_explanation = any("a0" in key.lower() for key in result.parameter_explanations.keys())
        validations.append(("A0 Explanation", has_a0_explanation, "Should have A0 explanation", f"Found: {has_a0_explanation}"))
    
    # Check for validation issues
    if expected.get("expect_validation_issues"):
        has_issues = len(result.potential_issues) > 0
        validations.append(("Validation Issues", has_issues, "Should detect validation issues", f"Found {len(result.potential_issues)} issues"))
    
    return validations    

# ========================================
# UPDATED MAIN TEST FUNCTION
# ========================================

async def test_unified_fits_agent_with_comprehensive_shared_parameters():
    """
    Enhanced version of test_unified_fits_agent_with_enhanced_cases
    with additional shared parameter test cases
    """
    print("Testing Unified FITS Agent v2.1 - Comprehensive Shared Parameters")
    print("=" * 80)
    
    agent = UnifiedFITSClassificationAgent()
    
    # Original test cases from test_unified_fits_agent_with_enhanced_cases
    original_test_cases = [
        # ... (ใช้ test cases เดิมจาก code ที่มีอยู่)
        {
            "name": "✅ Pure Statistics Analysis",
            "input": "Calculate mean, median, and standard deviation for my lightcurve data",
            "context": {"has_uploaded_files": True, "user_expertise": "intermediate"},
            "expected": {
                "primary_intent": "analysis",
                "analysis_types": ["statistics"],
                "routing_strategy": "analysis",
                "question_category": "unknown",  #  เพิ่มใหม่ใน v2.1
                "complexity_level": "intermediate",  #  เพิ่มใหม่ใน v2.1
                "has_parameters": True,
                "parameter_types": ["statistics"],
                "astrosage_required": False  #  เพิ่มเติม
            }
        },
        {
            "name": "✅ PSD Analysis with Custom Parameters",
            "input": "Compute power spectral density with frequency range 1e-4 to 0.1 Hz and 5000 bins",
            "context": {"has_uploaded_files": True, "user_expertise": "advanced"},
            "expected": {
                "primary_intent": "analysis",
                "analysis_types": ["psd"],
                "routing_strategy": "analysis",
                "question_category": "unknown",  #  ไม่มีคำถาม
                "complexity_level": "advanced",  #  จาก context
                "has_parameters": True,
                "parameter_types": ["psd"],
                "custom_params": {"low_freq": 1e-4, "high_freq": 0.1, "bins": 5000},
                "astrosage_required": False
            }
        },
        {
            "name": "✅ Power Law Fitting with Initial Guesses",
            "input": "Fit a power law model with initial amplitude of 2.5 and power index of 1.8",
            "context": {"has_uploaded_files": True, "user_expertise": "advanced"},
            "expected": {
                "primary_intent": "analysis",
                "analysis_types": ["fitting_power_law"],
                "routing_strategy": "analysis",
                "question_category": "unknown",
                "complexity_level": "advanced",
                "has_parameters": True,
                "parameter_types": ["fitting_power_law"],
                "custom_params": {"A0": 2.5, "b0": 1.8},
                "astrosage_required": False
            }
        },
        {
            "name": "✅ Pure General Question - Astronomy",
            "input": "What causes neutron stars to emit X-rays?",
            "context": {"user_expertise": "beginner"},
            "expected": {
                "primary_intent": "general",
                "analysis_types": [],
                "routing_strategy": "astrosage",
                "question_category": "astronomy",  #  เพิ่มใหม่ - คำถามดาราศาสตร์
                "complexity_level": "beginner",
                "has_parameters": False,
                "astrosage_required": True  #  สำคัญสำหรับ v2.1
            }
        },
        {
            "name": "✅ Data Analysis Question",
            "input": "What is power spectral density and how do I interpret the results?",
            "context": {"user_expertise": "intermediate"},
            "expected": {
                "primary_intent": "general",
                "analysis_types": [],
                "routing_strategy": "astrosage",
                "question_category": "data_analysis",  #  เพิ่มใหม่ - คำถามการวิเคราะห์ข้อมูล
                "complexity_level": "intermediate",
                "has_parameters": False,
                "astrosage_required": True
            }
        },
        {
            "name": "✅ Mixed Request - Question then Analysis",
            "input": "What is a bending power law? Then fit one to my data with break frequency around 0.01 Hz",
            "context": {"has_uploaded_files": True, "user_expertise": "intermediate"},
            "expected": {
                "primary_intent": "mixed",  #  สำคัญ - v2.1 ตรวจจับ mixed ได้ดีกว่า
                "analysis_types": ["fitting_bending_power_law"],
                "routing_strategy": "mixed",
                "question_category": "data_analysis",  #  เพิ่มใหม่ - เกี่ยวกับ bending power law
                "complexity_level": "intermediate",
                "has_parameters": True,
                "parameter_types": ["fitting_bending_power_law"],
                "custom_params": {"fb0": 0.01},
                "is_mixed_request": True,  #  เพิ่มเติม
                "astrosage_required": True  #  ต้องการ AstroSage เพื่ออธิบาย
            }
        },
        {
            "name": "✅ Complex Sequential Analysis",
            "input": "Perform complete timing analysis: compute PSD with 4000 bins, then fit both power law and bending power law models",
            "context": {"has_uploaded_files": True, "user_expertise": "expert"},
            "expected": {
                "primary_intent": "analysis",
                "analysis_types": ["psd", "fitting_power_law", "fitting_bending_power_law"],
                "routing_strategy": "analysis",
                "question_category": "unknown",  #  ไม่มีคำถาม แค่ analysis
                "complexity_level": "expert",  #  จาก context -> advanced
                "has_parameters": True,
                "parameter_types": ["psd", "fitting_power_law", "fitting_bending_power_law"],
                "custom_params": {"bins": 4000},
                "astrosage_required": False
            }
        },
        {
            "name": "✅ Beginner Mixed Request - Enhanced Detection",
            "input": "I'm new to X-ray astronomy. Calculate basic statistics for my neutron star data and explain what they mean",
            "context": {"has_uploaded_files": True, "user_expertise": "beginner"},
            "expected": {
                "primary_intent": "mixed",  #  v2.1 จับได้ดีกว่า
                "analysis_types": ["statistics"],
                "routing_strategy": "mixed",
                "question_category": "astronomy",  #  เพิ่มใหม่ - บริบท neutron star
                "complexity_level": "beginner",
                "has_parameters": True,
                "parameter_types": ["statistics"],
                "is_mixed_request": True,
                "astrosage_required": True  #  ต้องการอธิบาย
            }
        },
        {
            "name": "✅ Physics Question",
            "input": "Explain the physics behind X-ray emission from accreting black holes",
            "context": {"user_expertise": "advanced"},
            "expected": {
                "primary_intent": "general",
                "analysis_types": [],
                "routing_strategy": "astrosage",
                "question_category": "physics",  #  เพิ่มใหม่ - คำถามฟิสิกส์
                "complexity_level": "advanced",
                "has_parameters": False,
                "astrosage_required": True
            }
        },
        {
            "name": "✅ Methods Question",
            "input": "What's the best way to fit power law models to noisy data?",
            "context": {"user_expertise": "intermediate"},
            "expected": {
                "primary_intent": "general",
                "analysis_types": [],
                "routing_strategy": "astrosage",
                "question_category": "methods",  #  เพิ่มใหม่ - คำถามเกี่ยวกับวิธีการ
                "complexity_level": "intermediate",
                "has_parameters": False,
                "astrosage_required": True
            }
        },
        {
            "name": "✅ Advanced Mixed Request - Analysis + Interpretation",
            "input": "Calculate statistics and compute PSD, then explain what the frequency peaks might indicate about the neutron star's rotation",
            "context": {"has_uploaded_files": True, "user_expertise": "intermediate"},
            "expected": {
                "primary_intent": "mixed",
                "analysis_types": ["statistics", "psd"],
                "routing_strategy": "mixed",
                "question_category": "astronomy",  #  เพิ่มใหม่ - บริบท neutron star rotation
                "complexity_level": "intermediate",
                "has_parameters": True,
                "parameter_types": ["statistics", "psd"],
                "is_mixed_request": True,
                "astrosage_required": True
            }
        },
        {
            "name": "✅ Bending Power Law with Custom Parameters - Enhanced",
            "input": "Fit bending power law with shape parameter 2.0 and amplitude bounds from 1 to 100",
            "context": {"has_uploaded_files": True, "user_expertise": "advanced"},
            "expected": {
                "primary_intent": "analysis",
                "analysis_types": ["fitting_bending_power_law"],
                "routing_strategy": "analysis",
                "question_category": "unknown",
                "complexity_level": "advanced",
                "has_parameters": True,
                "parameter_types": ["fitting_bending_power_law"],
                "custom_params": {"sh0": 2.0, "A_min": 1.0, "A_max": 100.0},
                "astrosage_required": False
            }
        }
    ]
    
    # Additional test cases for shared parameters
    additional_test_cases = get_enhanced_test_cases_for_shared_parameters()
    
    # Combine all test cases
    all_test_cases = original_test_cases + additional_test_cases
    
    success_count = 0
    total_cost = 0.0
    total_time = 0.0
    
    for i, test_case in enumerate(all_test_cases, 1):
        print(f"\n{'='*90}")
        print(f"Test {i}: {test_case['name']}")
        print(f"Input: '{test_case['input']}'")
        print(f"Context: {test_case['context']}")
        print(f"{'='*90}")
        
        try:
            start_time = datetime.now()
            result = await agent.process_request(test_case["input"], test_case["context"])
            processing_time = (datetime.now() - start_time).total_seconds()
            
            total_cost += result.cost_estimate
            total_time += processing_time
            
            # Display results
            print(f"Processing successful in {processing_time:.3f}s")
            print(f"Cost: ${result.cost_estimate:.6f}")
            print(f"Confidence: {result.confidence:.2f}")
            
            print(f"\nClassification Results:")
            print(f"   Primary Intent: {result.primary_intent}")
            print(f"   Analysis Types: {result.analysis_types}")
            print(f"   Question Category: {result.question_category}")
            print(f"   Mixed Request: {result.is_mixed_request}")
            print(f"   Complexity Level: {result.complexity_level}")
            print(f"   Routing Strategy: {result.routing_strategy}")
            print(f"   AstroSage Required: {result.astrosage_required}")
            # print(f"   Custom Parameters: {result.parameters}")
            print(f"   Question Context: {result.question_context}")
            print(f"   Potential Issues: {len(result.potential_issues)} found")
            
            if result.parameters:
                print(f"\nParameters:")
                for analysis_type, params in result.parameters.items():
                    print(f"   {analysis_type}:")
                    for param_name, param_value in params.items():
                        print(f"      {param_name}: {param_value}")
            
            if result.parameter_explanations:
                print(f"\nParameter Explanations:")
                for key, explanation in result.parameter_explanations.items():
                    print(f"   {key}: {explanation}")
            
            # Enhanced validation including shared parameters
            expected = test_case["expected"]
            validations = []
            
            # Original validations
            intent_ok = result.primary_intent == expected["primary_intent"]
            validations.append(("Primary Intent", intent_ok, expected["primary_intent"], result.primary_intent))
            
            analysis_ok = set(result.analysis_types) == set(expected["analysis_types"])
            validations.append(("Analysis Types", analysis_ok, expected["analysis_types"], result.analysis_types))
            
            # Add shared parameter validations
            shared_validations = validate_shared_parameters(result, expected)
            validations.extend(shared_validations)
            
            print(f"\nValidation Results:")
            all_valid = True
            for check_name, is_valid, expected_val, actual_val in validations:
                status = "✅" if is_valid else "❌"
                print(f"   {check_name}: {status}")
                if not is_valid:
                    print(f"      Expected: {expected_val}")
                    print(f"      Got: {actual_val}")
                    all_valid = False
            
            if all_valid:
                success_count += 1
                print(f"\nPASS")
            else:
                print(f"\nFAIL")
            
        except Exception as e:
            print(f"Error: {str(e)}")
        
        await asyncio.sleep(0.3)
    
    # Final summary
    print(f"\n{'='*90}")
    print("COMPREHENSIVE SHARED PARAMETER TEST RESULTS")
    print(f"{'='*90}")
    
    print(f"Success Rate: {success_count}/{len(all_test_cases)} ({success_count/len(all_test_cases)*100:.1f}%)")
    print(f"Total Cost: ${total_cost:.6f}")
    print(f"Total Time: {total_time:.3f}s")
    
    return success_count == len(all_test_cases)

# ========================================
# FACTORY AND MAIN - FIXED VERSION
# ========================================

def create_unified_fits_agent(**kwargs) -> UnifiedFITSClassificationAgent:
    """Factory function to create Unified FITS Agent v2.1 - ORCHESTRATOR COMPATIBLE"""
    return UnifiedFITSClassificationAgent(**kwargs)


# Main execution - FIXED
if __name__ == "__main__":
    print(" Unified FITS Classification + Parameter Agent v2.1")
    print(" ORCHESTRATOR COMPATIBLE VERSION")
    print("=" * 60)

    async def main():
        """Main demo function - FIXED VERSION"""
        try:
            print("Choose option:")
            print("1. Test orchestrator compatibility")
            print("2. Run enhanced comprehensive tests") 
            print("3. Quick validation")
            print("4. Test comprehensive shared parameters")
            choice = input("Enter choice (1-4): ").strip()
            
            if choice == "1":
                success = await test_orchestrator_compatibility()
                if success:
                    print("\n🚀 READY FOR ORCHESTRATOR INTEGRATION!")
                else:
                    print("\n🔧 Additional fixes needed")
                    
            elif choice == "2":
                await test_unified_fits_agent_with_enhanced_cases()
                
            elif choice == "3":
                # Quick validation test - FIXED INDENTATION
                agent = UnifiedFITSClassificationAgent()
                
                print("\n🚀 Quick Validation Test")
                print("=" * 40)
                
                test_requests = [
                    ("Calculate statistics", {"has_uploaded_files": True}),
                    ("What is a neutron star?", {}),
                    ("What is PSD? Then compute it", {"has_uploaded_files": True})
                ]
                
                for i, (request, context) in enumerate(test_requests, 1):
                    print(f"\nTest {i}: '{request}'")
                    
                    try:
                        result = await agent.process_request(request, context)
                        
                        # Check orchestrator required attributes
                        required_attrs = ["question_category", "complexity_level", "primary_intent", "analysis_types", "confidence"]
                        missing = [attr for attr in required_attrs if not hasattr(result, attr) or getattr(result, attr) is None]
                        
                        print(f"   Intent: {result.primary_intent}")
                        print(f"   Question Category: {result.question_category}")
                        print(f"   Complexity: {result.complexity_level}")
                        print(f"   Orchestrator Compatible: {'✅' if not missing else '❌'}")
                        if missing:
                            print(f"   Missing: {missing}")
                            
                    except Exception as e:
                        print(f"   ❌ Test failed: {str(e)}")
                
                print(f"\n✅ Quick validation completed!")
                
            elif choice == "4":
                await test_unified_fits_agent_with_comprehensive_shared_parameters()
                
            else:
                print("Invalid choice, running enhanced comprehensive tests...")
                await test_unified_fits_agent_with_enhanced_cases()
            
        except Exception as e:
            print(f"\n❌ Execution failed: {str(e)}")
            import traceback
            traceback.print_exc()

    # Run the application
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Application interrupted")
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        print("\nMake sure you have:")
        print("   - OpenAI API key set (export OPENAI_API_KEY='your-key')")
        print("   - Required packages installed: pip install langchain-openai python-dotenv")
        print("   - Internet connection for OpenAI API")