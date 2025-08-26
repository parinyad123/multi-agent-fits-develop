#!/usr/bin/env python3
"""
Real AstroSage Integration for FITS Analysis Orchestrator
Connects to AstroSage-Llama-3.1-8B at 192.168.156.22:8080
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class AstroSageRequestType(Enum):
    """Types of AstroSage requests"""
    PURE_QUESTION = "pure_question"
    FITTING_INTERPRETATION = "fitting_interpretation"
    RESEARCH_CONSULTATION = "research_consultation"
    MIXED_CONTEXT = "mixed_context"


@dataclass
class AstroSageRequest:
    """AstroSage request structure"""
    request_type: AstroSageRequestType
    user_input: str
    question_category: str  # "astronomy", "physics", "data_analysis", "methods"
    complexity_level: str   # "beginner", "intermediate", "advanced"
    
    # Context information
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    analysis_results: Optional[Dict[str, Any]] = None
    user_expertise: str = "intermediate"
    session_context: Optional[Dict[str, Any]] = None
    
    # Metadata
    session_id: Optional[str] = None
    user_id: Optional[str] = None


@dataclass
class AstroSageResponse:
    """AstroSage response structure"""
    success: bool
    response_text: str
    confidence: float
    request_type: AstroSageRequestType
    
    # Metadata
    processing_time: float
    tokens_used: int
    cost_estimate: float
    model_used: str = "astrosage"
    
    # Enhanced information
    sources_mentioned: List[str] = field(default_factory=list)
    key_concepts: List[str] = field(default_factory=list)
    follow_up_suggestions: List[str] = field(default_factory=list)
    
    # Error information
    error: Optional[str] = None
    retry_suggested: bool = False


class ConversationManager:
    """Manages conversation history and summarization"""
    
    def __init__(self, max_history_length: int = 10):
        self.max_history_length = max_history_length
        self.summarization_threshold = 10
    
    def prepare_conversation_context(self, 
                                   conversation_history: List[Dict[str, str]],
                                   current_input: str,
                                   user_context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Prepare conversation context for AstroSage"""
        
        if len(conversation_history) <= self.summarization_threshold:
            # Use full history if short enough
            messages = conversation_history.copy()
        else:
            # Summarize older conversations, keep recent ones
            recent_messages = conversation_history[-5:]  # Last 5 messages
            older_messages = conversation_history[:-5]
            
            # Create summary of older messages
            summary = self._create_conversation_summary(older_messages, user_context)
            
            # Combine summary with recent messages
            messages = [
                {
                    "role": "user",
                    "content": f"PREVIOUS CONVERSATION SUMMARY:\n{summary}"
                }
            ] + recent_messages
        
        # Add current input
        messages.append({
            "role": "user", 
            "content": current_input
        })
        
        return messages
    
    def _create_conversation_summary(self, 
                                   messages: List[Dict[str, str]], 
                                   user_context: Dict[str, Any]) -> str:
        """Create intelligent summary of conversation history"""
        
        # Extract key information
        user_expertise = user_context.get("user_expertise", "intermediate")
        analysis_types = user_context.get("previous_analyses", [])
        
        # Analyze conversation themes
        themes = []
        if any("neutron star" in msg.get("content", "").lower() for msg in messages):
            themes.append("neutron star physics")
        if any("psd" in msg.get("content", "").lower() for msg in messages):
            themes.append("power spectral density analysis")
        if any("power law" in msg.get("content", "").lower() for msg in messages):
            themes.append("power law fitting")
        
        # Build summary
        summary_parts = []
        
        # User context
        summary_parts.append(f"The user is a {user_expertise}-level researcher.")
        
        # Discussion themes
        if themes:
            summary_parts.append(f"In our previous discussion, we covered: {', '.join(themes)}.")
        
        # Analysis context
        if analysis_types:
            summary_parts.append(f"The user has performed: {', '.join(analysis_types)} analysis.")
        
        # Key questions/topics from recent messages
        recent_topics = []
        for msg in messages[-3:]:  # Last 3 messages for key topics
            content = msg.get("content", "")
            if "?" in content:
                # Extract questions
                questions = [q.strip() for q in content.split("?") if q.strip()]
                recent_topics.extend(questions[:2])  # Max 2 questions
        
        if recent_topics:
            summary_parts.append(f"Recent topics: {'; '.join(recent_topics)}.")
        
        return " ".join(summary_parts)


class PromptTemplateManager:
    """Manages different prompt templates for different request types"""
    
    def __init__(self):
        self.templates = self._build_prompt_templates()
    
    def _build_prompt_templates(self) -> Dict[str, str]:
        """Build comprehensive prompt templates"""
        
        return {
            "system_base": """You are AstroSage, an expert astrophysicist AI assistant specializing in stellar physics, cosmology, observational astronomy, and X-ray astronomy. You provide accurate, detailed scientific explanations while maintaining conversation continuity. Always reference previous discussion points when relevant.

Your expertise includes:
- Stellar evolution and nucleosynthesis
- Compact objects (neutron stars, black holes, white dwarfs)  
- X-ray astronomy and high-energy astrophysics
- Timing analysis and power spectral density
- Statistical analysis of astronomical data
- Power law models in astrophysical contexts
- Observational techniques and data analysis

Always provide scientifically accurate information and cite relevant concepts when appropriate.""",

            "pure_question": """Focus on providing a comprehensive scientific explanation for the user's question. Consider their expertise level ({complexity_level}) and adjust the technical depth accordingly.

If this is a {question_category} question, ensure you cover the fundamental physics and observational aspects.""",

            "fitting_interpretation": """The user has performed data analysis and needs help interpreting their results. Here are their analysis results:

{analysis_results}

Please explain what these results mean in the context of astrophysical processes. Consider:
1. Physical interpretation of the fitted parameters
2. What the results tell us about the underlying source
3. Comparison with typical values in the literature
4. Potential implications for the astrophysical system
5. Suggestions for further analysis or investigation

Tailor your explanation to a {complexity_level} level user.""",

            "research_consultation": """The user is seeking research guidance. Consider their analysis history and current context:

User expertise: {user_expertise}
Previous analyses performed: {previous_analyses}
Current research context: {research_context}

Provide strategic advice on:
1. Next logical analysis steps
2. Parameter optimization suggestions
3. Complementary analysis techniques
4. Potential research directions
5. Relevant literature or methods to explore

Focus on actionable, scientifically sound recommendations.""",

            "mixed_context": """This is a mixed request combining questions and analysis. The user's analysis results are:

{analysis_results}

Address both the conceptual questions and provide context-aware interpretation of their results. Connect the theoretical concepts to their actual data analysis outcomes.""",

            "educational_beginner": """The user is new to this topic. Provide:
1. Clear, accessible explanations
2. Fundamental concepts and definitions
3. Real-world examples and analogies
4. Connections to basic physics principles
5. Suggestions for further learning

Avoid excessive jargon but maintain scientific accuracy.""",

            "educational_advanced": """The user has advanced knowledge. You can:
1. Use technical terminology appropriately
2. Reference current literature and research
3. Discuss cutting-edge developments
4. Provide detailed mathematical or physical insights
5. Suggest sophisticated analysis techniques

Assume familiarity with fundamental concepts."""
        }
    
    def build_prompt(self, request: AstroSageRequest, analysis_context: Optional[Dict] = None) -> List[Dict[str, str]]:
        """Build complete prompt for AstroSage request"""
        
        # Start with system message
        system_content = self.templates["system_base"]
        
        # Select appropriate template based on request type
        if request.request_type == AstroSageRequestType.PURE_QUESTION:
            template_key = "pure_question"
            if request.complexity_level == "beginner":
                template_key = "educational_beginner"
            elif request.complexity_level == "advanced":
                template_key = "educational_advanced"
        
        elif request.request_type == AstroSageRequestType.FITTING_INTERPRETATION:
            template_key = "fitting_interpretation"
        
        elif request.request_type == AstroSageRequestType.RESEARCH_CONSULTATION:
            template_key = "research_consultation"
        
        elif request.request_type == AstroSageRequestType.MIXED_CONTEXT:
            template_key = "mixed_context"
        
        else:
            template_key = "pure_question"
        
        # Build template context
        template_context = {
            "complexity_level": request.complexity_level,
            "question_category": request.question_category,
            "user_expertise": request.user_expertise,
            "analysis_results": self._format_analysis_results(request.analysis_results),
            "previous_analyses": getattr(request.session_context, 'previous_analyses', []),
            "research_context": getattr(request.session_context, 'research_context', 'General research')
        }
        
        # Format template
        user_prompt = self.templates[template_key].format(**template_context)
        
        # Build messages
        messages = [
            {"role": "system", "content": system_content + "\n\n" + user_prompt}
        ]
        
        return messages
    
    def _format_analysis_results(self, analysis_results: Optional[Dict[str, Any]]) -> str:
        """Format analysis results for inclusion in prompts"""
        
        if not analysis_results:
            return "No analysis results provided."
        
        formatted_parts = []
        
        for analysis_type, result in analysis_results.items():
            if analysis_type == "statistics":
                stats = result.get("results", {}).get("statistics", {})
                formatted_parts.append(f"Statistical Analysis:\n  Mean: {stats.get('mean', 'N/A')}\n  Standard Deviation: {stats.get('std', 'N/A')}\n  Median: {stats.get('median', 'N/A')}")
            
            elif analysis_type == "psd":
                psd_info = result.get("results", {})
                freq_range = psd_info.get("freq_range", {})
                formatted_parts.append(f"Power Spectral Density Analysis:\n  Frequency range: {freq_range.get('actual', 'N/A')} Hz\n  Number of frequency bins: {psd_info.get('n_points', 'N/A')}")
            
            elif "fitting" in analysis_type:
                fitted_params = result.get("results", {}).get("fitted_parameters", {})
                if analysis_type == "fitting_power_law":
                    formatted_parts.append(f"Power Law Fitting Results:\n  Amplitude (A): {fitted_params.get('A', 'N/A')}\n  Power law index (b): {fitted_params.get('b', 'N/A')}\n  Noise level (n): {fitted_params.get('n', 'N/A')}")
                elif analysis_type == "fitting_bending_power_law":
                    formatted_parts.append(f"Bending Power Law Fitting Results:\n  Amplitude (A): {fitted_params.get('A', 'N/A')}\n  Break frequency (fb): {fitted_params.get('fb', 'N/A')} Hz\n  Shape parameter (sh): {fitted_params.get('sh', 'N/A')}\n  Noise level (n): {fitted_params.get('n', 'N/A')}")
        
        return "\n\n".join(formatted_parts) if formatted_parts else "Analysis results are available but could not be formatted."


class RealAstroSageClient:
    """
    🤖 Real AstroSage Client
    
    Connects to AstroSage-Llama-3.1-8B at 192.168.156.22:8080
    Handles astronomy Q&A, fitting interpretation, and research consultation
    """
    
    def __init__(self, 
                 base_url: str = "http://192.168.156.22:8080",
                 timeout: int = 60,
                 max_retries: int = 3):
        self.base_url = base_url.rstrip('/')
        self.endpoint = f"{self.base_url}/v1/chat/completions"
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        
        # Initialize managers
        self.conversation_manager = ConversationManager()
        self.prompt_manager = PromptTemplateManager()
        
        # Default parameters
        self.default_params = {
            "model": "astrosage",
            "max_tokens": 600,
            "temperature": 0.2,
            "top_p": 0.95,
            "repeat_penalty": 1.05,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.05
        }
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "avg_response_time": 0.0,
            "request_types": {t.value: 0 for t in AstroSageRequestType}
        }
        
        logger.info(f"AstroSage client initialized: {self.endpoint}")
    
    async def process_request(self, request: AstroSageRequest) -> AstroSageResponse:
        """
        Main method to process AstroSage requests
        
        Args:
            request: AstroSageRequest with all context
            
        Returns:
            AstroSageResponse with results and metadata
        """
        start_time = datetime.now()
        self.stats["total_requests"] += 1
        self.stats["request_types"][request.request_type.value] += 1
        
        try:
            # Prepare conversation context
            conversation_context = self.conversation_manager.prepare_conversation_context(
                request.conversation_history,
                request.user_input,
                {
                    "user_expertise": request.user_expertise,
                    "previous_analyses": getattr(request.session_context, 'previous_analyses', [])
                }
            )
            
            # Build prompt
            system_messages = self.prompt_manager.build_prompt(request, request.analysis_results)
            
            # Combine system prompt with conversation
            messages = system_messages + conversation_context
            
            # Make API call with retries
            response_data = await self._make_api_call_with_retry(messages)
            
            # Parse response
            response_text = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Extract metadata
            usage = response_data.get("usage", {})
            tokens_used = usage.get("total_tokens", 0)
            cost_estimate = self._estimate_cost(tokens_used)
            
            # Build response
            processing_time = (datetime.now() - start_time).total_seconds()
            
            astro_response = AstroSageResponse(
                success=True,
                response_text=response_text,
                confidence=0.9,  # AstroSage is generally confident
                request_type=request.request_type,
                processing_time=processing_time,
                tokens_used=tokens_used,
                cost_estimate=cost_estimate,
                model_used="astrosage"
            )
            
            # Enhance response with extracted information
            astro_response = self._enhance_response(astro_response, response_text)
            
            # Update statistics
            self.stats["successful_requests"] += 1
            self.stats["total_tokens"] += tokens_used
            self.stats["total_cost"] += cost_estimate
            self._update_avg_response_time(processing_time)
            
            logger.info(f"AstroSage request completed in {processing_time:.2f}s ({tokens_used} tokens)")
            
            return astro_response
            
        except Exception as e:
            self.stats["failed_requests"] += 1
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.error(f"AstroSage request failed: {str(e)}")
            
            return AstroSageResponse(
                success=False,
                response_text="",
                confidence=0.0,
                request_type=request.request_type,
                processing_time=processing_time,
                tokens_used=0,
                cost_estimate=0.0,
                error=str(e),
                retry_suggested=self._should_retry_error(str(e))
            )
    
    async def _make_api_call_with_retry(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Make API call with retry logic"""
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession(timeout=self.timeout) as session:
                    payload = {
                        **self.default_params,
                        "messages": messages
                    }
                    
                    async with session.post(
                        self.endpoint,
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    ) as response:
                        
                        if response.status == 200:
                            return await response.json()
                        else:
                            error_text = await response.text()
                            raise aiohttp.ClientResponseError(
                                request_info=response.request_info,
                                history=response.history,
                                status=response.status,
                                message=f"HTTP {response.status}: {error_text}"
                            )
            
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"AstroSage API call failed (attempt {attempt + 1}), retrying in {wait_time}s: {str(e)}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"AstroSage API call failed after {self.max_retries} attempts: {str(e)}")
        
        raise last_error
    
    def _enhance_response(self, response: AstroSageResponse, response_text: str) -> AstroSageResponse:
        """Enhance response with extracted information"""
        
        # Extract sources mentioned (simple keyword detection)
        sources = []
        source_keywords = ["paper", "study", "research", "observation", "survey", "mission", "telescope"]
        for keyword in source_keywords:
            if keyword in response_text.lower():
                sources.append(keyword)
        response.sources_mentioned = list(set(sources))
        
        # Extract key concepts
        concepts = []
        concept_keywords = [
            "neutron star", "black hole", "white dwarf", "accretion", "emission",
            "power law", "break frequency", "variability", "timing", "spectrum"
        ]
        for concept in concept_keywords:
            if concept in response_text.lower():
                concepts.append(concept)
        response.key_concepts = list(set(concepts))
        
        # Generate follow-up suggestions based on content
        follow_ups = []
        if "power law" in response_text.lower():
            follow_ups.append("Consider comparing with bending power law models")
        if "frequency" in response_text.lower():
            follow_ups.append("Analyze frequency-dependent features")
        if "variability" in response_text.lower():
            follow_ups.append("Investigate timing analysis techniques")
        
        response.follow_up_suggestions = follow_ups[:3]  # Limit to 3
        
        return response
    
    def _estimate_cost(self, tokens_used: int) -> float:
        """Estimate cost based on token usage"""
        # Rough estimate - adjust based on actual pricing
        return tokens_used * 0.000002  # $0.002 per 1K tokens
    
    def _should_retry_error(self, error_message: str) -> bool:
        """Determine if error should trigger retry suggestion"""
        retry_indicators = ["timeout", "connection", "network", "503", "502", "500"]
        return any(indicator in error_message.lower() for indicator in retry_indicators)
    
    def _update_avg_response_time(self, processing_time: float):
        """Update average response time"""
        total_requests = self.stats["successful_requests"]
        if total_requests > 0:
            current_avg = self.stats["avg_response_time"]
            self.stats["avg_response_time"] = (
                (current_avg * (total_requests - 1) + processing_time) / total_requests
            )
    
    # ========================================
    # CONVENIENCE METHODS FOR ORCHESTRATOR
    # ========================================
    
    async def answer_question(self, 
                            question: str,
                            question_category: str = "astronomy",
                            complexity_level: str = "intermediate",
                            conversation_history: List[Dict[str, str]] = None,
                            session_context: Dict[str, Any] = None) -> AstroSageResponse:
        """Convenience method for pure questions"""
        
        request = AstroSageRequest(
            request_type=AstroSageRequestType.PURE_QUESTION,
            user_input=question,
            question_category=question_category,
            complexity_level=complexity_level,
            conversation_history=conversation_history or [],
            session_context=session_context
        )
        
        return await self.process_request(request)
    
    async def interpret_fitting_results(self,
                                      user_input: str,
                                      analysis_results: Dict[str, Any],
                                      complexity_level: str = "intermediate",
                                      conversation_history: List[Dict[str, str]] = None,
                                      session_context: Dict[str, Any] = None) -> AstroSageResponse:
        """Convenience method for fitting interpretation"""
        
        request = AstroSageRequest(
            request_type=AstroSageRequestType.FITTING_INTERPRETATION,
            user_input=user_input,
            question_category="data_analysis",
            complexity_level=complexity_level,
            analysis_results=analysis_results,
            conversation_history=conversation_history or [],
            session_context=session_context
        )
        
        return await self.process_request(request)
    
    async def provide_research_consultation(self,
                                          user_input: str,
                                          user_expertise: str = "intermediate",
                                          previous_analyses: List[str] = None,
                                          conversation_history: List[Dict[str, str]] = None,
                                          session_context: Dict[str, Any] = None) -> AstroSageResponse:
        """Convenience method for research consultation"""
        
        if session_context is None:
            session_context = {}
        session_context['previous_analyses'] = previous_analyses or []
        
        request = AstroSageRequest(
            request_type=AstroSageRequestType.RESEARCH_CONSULTATION,
            user_input=user_input,
            question_category="methods",
            complexity_level=user_expertise,
            user_expertise=user_expertise,
            conversation_history=conversation_history or [],
            session_context=session_context
        )
        
        return await self.process_request(request)
    
    # ========================================
    # UTILITY AND MONITORING METHODS
    # ========================================
    
    async def health_check(self) -> Dict[str, Any]:
        """Check AstroSage service health"""
        try:
            test_messages = [
                {"role": "system", "content": "You are AstroSage, an astrophysics AI assistant."},
                {"role": "user", "content": "Hello"}
            ]
            
            start_time = datetime.now()
            response_data = await self._make_api_call_with_retry(test_messages)
            response_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "status": "healthy",
                "response_time": response_time,
                "endpoint": self.endpoint,
                "model": "astrosage"
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "endpoint": self.endpoint
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive client statistics"""
        total_requests = max(self.stats["total_requests"], 1)
        
        return {
            "requests": {
                "total": self.stats["total_requests"],
                "successful": self.stats["successful_requests"],
                "failed": self.stats["failed_requests"],
                "success_rate": self.stats["successful_requests"] / total_requests
            },
            "performance": {
                "avg_response_time": self.stats["avg_response_time"],
                "total_tokens": self.stats["total_tokens"],
                "total_cost": self.stats["total_cost"],
                "avg_tokens_per_request": self.stats["total_tokens"] / total_requests,
                "avg_cost_per_request": self.stats["total_cost"] / total_requests
            },
            "request_types": self.stats["request_types"],
            "endpoint": self.endpoint
        }


# ========================================
# FACTORY FUNCTION
# ========================================

def create_astrosage_client(base_url: str = "http://192.168.156.22:8080") -> RealAstroSageClient:
    """Factory function to create AstroSage client"""
    return RealAstroSageClient(base_url=base_url)


# ========================================
# INTEGRATION WITH ORCHESTRATOR
# ========================================

async def integrate_with_orchestrator_example():
    """Example of how to integrate with orchestrator"""
    
    # Replace mock implementation in orchestrator
    async def _execute_astrosage_task(self, 
                                    task: OrchestratorTask,
                                    user_input: str,
                                    context_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute AstroSage task using real client"""
        
        # Create real client
        astrosage_client = create_astrosage_client()
        
        # Determine request type based on task parameters
        task_params = task.parameters
        question_category = task_params.get("question_category", "astronomy")
        complexity_level = task_params.get("complexity_level", "intermediate")
        
        # Check if we have analysis context
        if task_params.get("analysis_context") and context_results:
            # Fitting interpretation request
            response = await astrosage_client.interpret_fitting_results(
                user_input=user_input,
                analysis_results=context_results,
                complexity_level=complexity_level
            )
        else:
            # Pure question request
            response = await astrosage_client.answer_question(
                question=user_input,
                question_category=question_category,
                complexity_level=complexity_level
            )
        
        # Convert to orchestrator format
        return {
            "response": response.response_text,
            "confidence": response.confidence,
            "sources": response.sources_mentioned,
            "key_concepts": response.key_concepts,
            "follow_up_suggestions": response.follow_up_suggestions,
            "processing_time": response.processing_time,
            "tokens_used": response.tokens_used,
            "cost": response.cost_estimate,
            "success": response.success,
            "error": response.error
        }


# ========================================
# TESTING AND VALIDATION
# ========================================

async def test_astrosage_integration():
    """Comprehensive test suite for AstroSage integration"""
    
    print("🤖 Testing Real AstroSage Integration")
    print("=" * 60)
    
    client = create_astrosage_client()
    
    # Test cases covering different scenarios
    test_cases = [
        {
            "name": "Pure Astronomy Question",
            "type": "question",
            "input": "What is a neutron star and how does it form?",
            "params": {
                "question_category": "astronomy",
                "complexity_level": "intermediate"
            }
        },
        {
            "name": "Data Analysis Question",
            "type": "question", 
            "input": "What is power spectral density and how is it used in X-ray astronomy?",
            "params": {
                "question_category": "data_analysis",
                "complexity_level": "beginner"
            }
        },
        {
            "name": "Fitting Interpretation",
            "type": "interpretation",
            "input": "What do these power law fitting results tell me about my neutron star data?",
            "analysis_results": {
                "fitting_power_law": {
                    "results": {
                        "fitted_parameters": {"A": 1.23, "b": 1.45, "n": 0.01},
                        "file_name": "neutron_star_data.fits"
                    }
                }
            },
            "params": {
                "complexity_level": "intermediate"
            }
        },
        {
            "name": "Research Consultation", 
            "type": "consultation",
            "input": "I'm studying neutron star variability. What analysis should I do next?",
            "params": {
                "user_expertise": "advanced",
                "previous_analyses": ["statistics", "psd", "fitting_power_law"]
            }
        },
        {
            "name": "Complex Mixed Request",
            "type": "question",
            "input": "Explain the physics behind power law noise in accreting systems",
            "params": {
                "question_category": "physics",
                "complexity_level": "advanced"
            }
        }
    ]
    
    successful_tests = 0
    total_cost = 0.0
    total_time = 0.0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}: {test_case['name']}")
        print(f"Input: '{test_case['input']}'")
        print(f"Type: {test_case['type']}")
        print(f"{'='*80}")
        
        try:
            start_time = datetime.now()
            
            # Execute based on test type
            if test_case["type"] == "question":
                response = await client.answer_question(
                    question=test_case["input"],
                    **test_case["params"]
                )
            
            elif test_case["type"] == "interpretation":
                response = await client.interpret_fitting_results(
                    user_input=test_case["input"],
                    analysis_results=test_case["analysis_results"],
                    **test_case["params"]
                )
            
            elif test_case["type"] == "consultation":
                response = await client.provide_research_consultation(
                    user_input=test_case["input"],
                    **test_case["params"]
                )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            total_time += processing_time
            total_cost += response.cost_estimate
            
            # Display results
            print(f"✅ Success: {response.success}")
            print(f"⏱️  Processing Time: {response.processing_time:.3f}s")
            print(f"🎯 Confidence: {response.confidence:.2f}")
            print(f"💰 Cost: ${response.cost_estimate:.6f}")
            print(f"🔤 Tokens: {response.tokens_used}")
            
            if response.success:
                print(f"\n📝 Response Preview:")
                print(f"   {response.response_text[:200]}...")
                
                if response.key_concepts:
                    print(f"\n🔑 Key Concepts: {', '.join(response.key_concepts)}")
                
                if response.follow_up_suggestions:
                    print(f"\n💡 Follow-up Suggestions:")
                    for suggestion in response.follow_up_suggestions:
                        print(f"   • {suggestion}")
                
                successful_tests += 1
            else:
                print(f"\n❌ Error: {response.error}")
        
        except Exception as e:
            print(f"❌ Test failed with exception: {str(e)}")
        
        await asyncio.sleep(1)  # Rate limiting
    
    # Final summary
    print(f"\n{'='*80}")
    print("🎯 ASTROSAGE INTEGRATION TEST RESULTS")
    print(f"{'='*80}")
    
    print(f"✅ Success Rate: {successful_tests}/{len(test_cases)} ({successful_tests/len(test_cases)*100:.1f}%)")
    print(f"💰 Total Cost: ${total_cost:.6f}")
    print(f"⏱️  Total Time: {total_time:.3f}s")
    print(f"📊 Avg Cost per Request: ${total_cost/len(test_cases):.6f}")
    print(f"📈 Avg Time per Request: {total_time/len(test_cases):.3f}s")
    
    # Show client statistics
    stats = client.get_stats()
    print(f"\n📊 CLIENT STATISTICS:")
    print(f"   Success Rate: {stats['requests']['success_rate']*100:.1f}%")
    print(f"   Avg Response Time: {stats['performance']['avg_response_time']:.3f}s")
    print(f"   Total Tokens: {stats['performance']['total_tokens']}")
    print(f"   Request Types: {stats['request_types']}")
    
    # Health check
    health = await client.health_check()
    print(f"\n🏥 HEALTH CHECK:")
    print(f"   Status: {health['status']}")
    if health['status'] == 'healthy':
        print(f"   Response Time: {health['response_time']:.3f}s")
    else:
        print(f"   Error: {health.get('error', 'Unknown')}")
    
    if successful_tests == len(test_cases):
        print(f"\n🎉 ALL TESTS PASSED! AstroSage integration is ready!")
    else:
        print(f"\n🟡 Some tests failed. Check AstroSage service availability.")
    
    return successful_tests == len(test_cases)


# ========================================
# ORCHESTRATOR INTEGRATION EXAMPLE
# ========================================

class AstroSageOrchestrationHelper:
    """Helper class for integrating AstroSage with orchestrator"""
    
    def __init__(self, astrosage_client: RealAstroSageClient):
        self.client = astrosage_client
    
    async def execute_astrosage_task(self, 
                                   task_params: Dict[str, Any],
                                   user_input: str,
                                   context_results: Dict[str, Any] = None,
                                   conversation_history: List[Dict[str, str]] = None,
                                   session_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute AstroSage task for orchestrator
        
        This method replaces the mock implementation in orchestrator
        """
        
        question_category = task_params.get("question_category", "astronomy")
        complexity_level = task_params.get("complexity_level", "intermediate") 
        context_type = task_params.get("context", "standalone")
        
        try:
            # Determine request type and execute
            if context_results and task_params.get("analysis_context"):
                # Fitting interpretation with analysis context
                response = await self.client.interpret_fitting_results(
                    user_input=user_input,
                    analysis_results=context_results,
                    complexity_level=complexity_level,
                    conversation_history=conversation_history,
                    session_context=session_context
                )
            
            elif context_type == "mixed_request":
                # Mixed request handling
                request = AstroSageRequest(
                    request_type=AstroSageRequestType.MIXED_CONTEXT,
                    user_input=user_input,
                    question_category=question_category,
                    complexity_level=complexity_level,
                    conversation_history=conversation_history or [],
                    analysis_results=context_results,
                    session_context=session_context
                )
                response = await self.client.process_request(request)
            
            elif question_category == "methods" or "research" in user_input.lower():
                # Research consultation
                user_expertise = session_context.get("user_expertise", "intermediate") if session_context else "intermediate"
                previous_analyses = session_context.get("previous_analyses", []) if session_context else []
                
                response = await self.client.provide_research_consultation(
                    user_input=user_input,
                    user_expertise=user_expertise,
                    previous_analyses=previous_analyses,
                    conversation_history=conversation_history,
                    session_context=session_context
                )
            
            else:
                # Pure question
                response = await self.client.answer_question(
                    question=user_input,
                    question_category=question_category,
                    complexity_level=complexity_level,
                    conversation_history=conversation_history,
                    session_context=session_context
                )
            
            # Convert to orchestrator format
            return {
                "response": response.response_text,
                "confidence": response.confidence,
                "sources": response.sources_mentioned,
                "key_concepts": response.key_concepts,
                "follow_up_suggestions": response.follow_up_suggestions,
                "processing_time": response.processing_time,
                "tokens_used": response.tokens_used,
                "cost": response.cost_estimate,
                "success": response.success,
                "error": response.error,
                "model_used": response.model_used
            }
            
        except Exception as e:
            logger.error(f"AstroSage task execution failed: {str(e)}")
            return {
                "response": "",
                "confidence": 0.0,
                "sources": [],
                "key_concepts": [],
                "follow_up_suggestions": [],
                "processing_time": 0.0,
                "tokens_used": 0,
                "cost": 0.0,
                "success": False,
                "error": str(e),
                "model_used": "astrosage"
            }


# ========================================
# PRODUCTION DEPLOYMENT CONSIDERATIONS  
# ========================================

class ProductionAstroSageClient(RealAstroSageClient):
    """Production-ready AstroSage client with enhanced features"""
    
    def __init__(self, 
                 base_url: str = "http://192.168.156.22:8080",
                 timeout: int = 60,
                 max_retries: int = 3,
                 enable_caching: bool = True,
                 cache_ttl: int = 300):  # 5 minutes
        super().__init__(base_url, timeout, max_retries)
        
        # Production features
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        self.response_cache: Dict[str, Dict] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # Enhanced monitoring
        self.error_counts: Dict[str, int] = {}
        self.response_times: List[float] = []
        self.max_response_time_history = 100
    
    async def process_request(self, request: AstroSageRequest) -> AstroSageResponse:
        """Enhanced request processing with caching and monitoring"""
        
        # Check cache first
        if self.enable_caching:
            cache_key = self._generate_cache_key(request)
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                logger.debug(f"Cache hit for AstroSage request: {request.user_input[:50]}...")
                return cached_response
        
        # Process request normally
        response = await super().process_request(request)
        
        # Cache successful responses
        if self.enable_caching and response.success:
            cache_key = self._generate_cache_key(request)
            self._cache_response(cache_key, response)
        
        # Update monitoring data
        self._update_monitoring_data(response)
        
        return response
    
    def _generate_cache_key(self, request: AstroSageRequest) -> str:
        """Generate cache key for request"""
        key_data = {
            "type": request.request_type.value,
            "input": request.user_input,
            "category": request.question_category,
            "complexity": request.complexity_level,
            "has_analysis": bool(request.analysis_results)
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[AstroSageResponse]:
        """Get cached response if still valid"""
        if cache_key not in self.response_cache:
            return None
        
        cache_time = self.cache_timestamps.get(cache_key)
        if not cache_time or (datetime.now() - cache_time).total_seconds() > self.cache_ttl:
            # Cache expired
            self.response_cache.pop(cache_key, None)
            self.cache_timestamps.pop(cache_key, None)
            return None
        
        cached_data = self.response_cache[cache_key]
        return AstroSageResponse(**cached_data)
    
    def _cache_response(self, cache_key: str, response: AstroSageResponse):
        """Cache response"""
        # Convert response to dict for caching
        response_dict = {
            "success": response.success,
            "response_text": response.response_text,
            "confidence": response.confidence,
            "request_type": response.request_type,
            "processing_time": 0.001,  # Cached responses are instant
            "tokens_used": response.tokens_used,
            "cost_estimate": 0.0,  # No cost for cached responses
            "model_used": response.model_used,
            "sources_mentioned": response.sources_mentioned,
            "key_concepts": response.key_concepts,
            "follow_up_suggestions": response.follow_up_suggestions
        }
        
        self.response_cache[cache_key] = response_dict
        self.cache_timestamps[cache_key] = datetime.now()
    
    def _update_monitoring_data(self, response: AstroSageResponse):
        """Update monitoring data"""
        # Track response times
        self.response_times.append(response.processing_time)
        if len(self.response_times) > self.max_response_time_history:
            self.response_times.pop(0)
        
        # Track errors
        if not response.success and response.error:
            error_type = response.error.split(":")[0]  # Get error type
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
    
    def get_production_stats(self) -> Dict[str, Any]:
        """Get production-specific statistics"""
        base_stats = self.get_stats()
        
        # Add cache statistics
        cache_stats = {
            "cache_enabled": self.enable_caching,
            "cache_size": len(self.response_cache),
            "cache_hit_rate": 0.0  # Would need to track hits/misses
        }
        
        # Add monitoring data
        monitoring_stats = {
            "error_breakdown": self.error_counts,
            "response_time_p95": self._calculate_percentile(self.response_times, 95) if self.response_times else 0.0,
            "response_time_p99": self._calculate_percentile(self.response_times, 99) if self.response_times else 0.0
        }
        
        return {
            **base_stats,
            "cache": cache_stats,
            "monitoring": monitoring_stats
        }
    
    def _calculate_percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of response times"""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int((percentile / 100) * len(sorted_values))
        return sorted_values[min(index, len(sorted_values) - 1)]