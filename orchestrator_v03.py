#!/usr/bin/env python3
"""
orchestrator_v03.py
====================
FITS Analysis Orchestrator
Coordinates between Unified Classification Agent, FITS Analysis Tools, and AstroSage
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib

# External service imports (will be implemented separately)
# from unified_fits_classification_agent import UnifiedFITSClassificationAgent, UnifiedFITSResult
from unified_FITS_classification_parameter_agent_v02_3 import UnifiedFITSClassificationAgent, UnifiedFITSResult
from real_astrosage_client import create_astrosage_client, AstroSageOrchestrationHelper

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RequestType(Enum):
    ANALYSIS = "analysis"
    GENERAL = "general" 
    MIXED = "mixed"


class ExecutionStrategy(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"


@dataclass
class SessionContext:
    """Complete session context information"""
    session_id: str
    user_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    
    # User profile
    user_expertise: str = "intermediate"  # beginner, intermediate, advanced
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    
    # File context
    uploaded_files: List[str] = field(default_factory=list)
    current_file: Optional[str] = None
    
    # Analysis history
    analysis_history: List[Dict[str, Any]] = field(default_factory=list)
    parameter_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Session state
    conversation_count: int = 0
    total_cost: float = 0.0
    
    # Context for current request
    current_analysis_types: List[str] = field(default_factory=list)
    previous_results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestratorTask:
    """Individual task within workflow"""
    task_id: str
    task_type: str  # "classification", "analysis", "astrosage"
    agent: str      # Agent responsible for this task
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)  # Task IDs this depends on
    status: str = "pending"  # pending, running, completed, failed
    result: Any = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


@dataclass
class WorkflowPlan:
    """Complete workflow execution plan"""
    workflow_id: str
    request_type: RequestType
    execution_strategy: ExecutionStrategy
    tasks: List[OrchestratorTask] = field(default_factory=list)
    estimated_time: float = 0.0
    estimated_cost: float = 0.0
    
    # Execution state
    status: str = "planned"  # planned, executing, completed, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    actual_cost: float = 0.0


@dataclass
class OrchestratorResponse:
    """Final orchestrated response"""
    request_id: str
    session_id: str
    success: bool
    
    # Classification info
    primary_intent: str
    analysis_types: List[str]
    confidence: float
    
    # Results
    analysis_results: Dict[str, Any] = field(default_factory=dict)
    astrosage_response: Optional[str] = None
    
    # Execution metadata
    workflow: Optional[WorkflowPlan] = None
    processing_time: float = 0.0
    total_cost: float = 0.0
    
    # Enhanced response
    explanation: Optional[str] = None
    suggested_next_steps: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Context updates
    session_context: Optional[SessionContext] = None


class FITSAnalysisOrchestrator:
    """
    🎭 FITS Analysis Orchestrator
    
    Coordinates between:
    - Unified Classification Agent (intent + parameters)
    - FITS Analysis Tools (statistics, PSD, fitting) 
    - AstroSage Client (astronomy Q&A)
    
    Manages sessions, workflows, and response aggregation.
    """
    
    def __init__(self):
        self.orchestrator_id = str(uuid.uuid4())
        self.logger = logging.getLogger(f"orchestrator.{self.orchestrator_id}")
        
        # Initialize agents and services
        self.classification_agent = UnifiedFITSClassificationAgent()
        # self.fits_service = FITSAnalysisService()  # Will implement
        # self.astrosage_client = AstroSageClient()  # Will implement
        
        # Session management
        self.active_sessions: Dict[str, SessionContext] = {}
        self.session_ttl = timedelta(hours=4)  # 4 hour session timeout
        
        # Workflow management
        self.active_workflows: Dict[str, WorkflowPlan] = {}
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_sessions": 0,
            "total_cost": 0.0,
            "avg_processing_time": 0.0,
            "request_types": {"analysis": 0, "general": 0, "mixed": 0}
        }
        
        self.logger.info(f"FITS Orchestrator initialized: {self.orchestrator_id}")

        self.astrosage_client = create_astrosage_client()
        self.astrosage_helper = AstroSageOrchestrationHelper(self.astrosage_client)
    
    async def process_request(self, 
                            user_input: str,
                            session_id: Optional[str] = None,
                            user_id: Optional[str] = None,
                            context: Optional[Dict[str, Any]] = None) -> OrchestratorResponse:
        """
        Main orchestrator method - processes complete user request
        
        Args:
            user_input: User's natural language request
            session_id: Optional session ID (creates new if None)
            user_id: Optional user ID for personalization
            context: Additional context (files, preferences, etc.)
            
        Returns:
            Complete orchestrated response
        """
        request_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        self.stats["total_requests"] += 1
        self.logger.info(f"Processing request {request_id}: '{user_input[:50]}...'")
        
        try:
            # 1. Session Management
            if user_input.strip() == "":
                raise ValueError("Empty user input")

            session_context = await self._get_or_create_session(session_id, user_id, context)
            session_context.conversation_count += 1
            session_context.last_activity = datetime.now()
            
            # 2. Intent Classification & Parameter Extraction
            try:
                classification_result = await asyncio.wait_for(
                    self._classify_and_extract_parameters(user_input, session_context),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                raise Exception("Classification timeout - please try again")
            
            # Validate classification result
            if not classification_result or not hasattr(classification_result, 'primary_intent'):
                raise Exception("Invalid classification result")
            
            # 3. Workflow Planning
            workflow_plan = await self._plan_workflow(
                classification_result, session_context
            )
            
            # 4. Workflow Execution
            execution_result = await self._execute_workflow(
                workflow_plan, user_input, session_context
            )
            
            # 5. Response Assembly
            response = await self._assemble_response(
                request_id, classification_result, execution_result, 
                workflow_plan, session_context, start_time
            )
            
            # 6. Session State Update
            await self._update_session_state(session_context, classification_result, response)
            
            self.stats["successful_requests"] += 1
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Request {request_id} completed successfully in {processing_time:.3f}s")
            
            return response

        except ValueError as e:
            self.logger.error(f"Validation error for request {request_id}: {str(e)}")
            return await self._create_error_response(request_id, session_id or "unknown", f"Input validation failed: {str(e)}", start_time)
        
        except asyncio.TimeoutError as e:
            self.logger.error(f"Timeout error for request {request_id}: {str(e)}")
            return await self._create_error_response(request_id, session_id or "unknown", "Request timed out", start_time)
       
        except Exception as e:
            self.stats["failed_requests"] += 1
            self.logger.error(f"Request {request_id} failed: {str(e)}")
            
            # Create error response
            return await self._create_error_response(
                request_id, session_id or "unknown", str(e), start_time
            )
    
    async def _get_or_create_session(self, 
                                   session_id: Optional[str], 
                                   user_id: Optional[str],
                                   context: Optional[Dict[str, Any]]) -> SessionContext:
        """Get existing session or create new one"""
        
        # Clean up expired sessions first
        await self._cleanup_expired_sessions()
        
        if session_id and session_id in self.active_sessions:
            session_context = self.active_sessions[session_id]
            self.logger.debug(f"Retrieved existing session: {session_id}")
        else:
            # Create new session
            session_id = session_id or str(uuid.uuid4())
            session_context = SessionContext(
                session_id=session_id,
                user_id=user_id
            )
            
            # Apply context if provided
            if context:
                if "uploaded_files" in context:
                    session_context.uploaded_files = context["uploaded_files"]
                if "user_expertise" in context:
                    session_context.user_expertise = context["user_expertise"]
                if "user_preferences" in context:
                    session_context.user_preferences = context["user_preferences"]
            
            self.active_sessions[session_id] = session_context
            self.stats["total_sessions"] += 1
            
            self.logger.info(f"Created new session: {session_id}")
        
        return session_context
    
    async def _classify_and_extract_parameters(self, 
                                             user_input: str, 
                                             session_context: SessionContext) -> UnifiedFITSResult:
        """Use unified agent for classification and parameter extraction"""
        
        # Build context for classification agent
        classification_context = {
            "has_uploaded_files": bool(session_context.uploaded_files),
            "user_expertise": session_context.user_expertise,
            "previous_analyses": [item.get("analysis_type") for item in session_context.analysis_history[-3:]],
            "conversation_count": session_context.conversation_count,
            "current_file": session_context.current_file,
            "user_preferences": session_context.user_preferences
        }
        
        # Add user preferences
        if session_context.user_preferences:
            classification_context.update(session_context.user_preferences)
        
        # self.logger.debug(f"Calling classification agent with context: {classification_context}")
        self.logger.debug(f"Classification context: {classification_context}")

        # Call unified classification agent
        result = await self.classification_agent.process_request(user_input, classification_context)
        
        # Update session context with current analysis types
        session_context.current_analysis_types = result.analysis_types
        
        # Track request type statistics
        self.stats["request_types"][result.primary_intent] += 1
        
        self.logger.info(f"Classification completed: {result.primary_intent} → {result.analysis_types} "
                        f"(confidence: {result.confidence:.2f})")

        # Validate result has required orchestrator attributes
        required_attrs = ['question_category', 'complexity_level', 'primary_intent']
        missing_attrs = [attr for attr in required_attrs if not hasattr(result, attr)]
        if missing_attrs:
            self.logger.warning(f"Classification result missing attributes: {missing_attrs}")
        
        return result
    
    async def _plan_workflow(self, 
                           classification_result: UnifiedFITSResult,
                           session_context: SessionContext) -> WorkflowPlan:
        """Plan workflow based on classification result"""
        
        workflow_id = str(uuid.uuid4())
        request_type = RequestType(classification_result.primary_intent)
        
        # Determine execution strategy
        if classification_result.is_mixed_request:
            if classification_result.question_context == "before_analysis":
                execution_strategy = ExecutionStrategy.SEQUENTIAL
            elif classification_result.question_context == "after_analysis":
                execution_strategy = ExecutionStrategy.SEQUENTIAL  
            else:  # parallel
                execution_strategy = ExecutionStrategy.PARALLEL
        else:
            execution_strategy = ExecutionStrategy.SEQUENTIAL
        
        workflow_plan = WorkflowPlan(
            workflow_id=workflow_id,
            request_type=request_type,
            execution_strategy=execution_strategy
        )
        
        # Build task list based on intent
        if request_type == RequestType.ANALYSIS:
            workflow_plan.tasks = self._create_analysis_tasks(classification_result)
            
        elif request_type == RequestType.GENERAL:
            workflow_plan.tasks = self._create_general_tasks(classification_result)
            
        elif request_type == RequestType.MIXED:
            workflow_plan.tasks = self._create_mixed_tasks(classification_result)
        
        # Estimate time and cost
        workflow_plan.estimated_time = self._estimate_workflow_time(workflow_plan.tasks)
        workflow_plan.estimated_cost = self._estimate_workflow_cost(workflow_plan.tasks)
        
        self.active_workflows[workflow_id] = workflow_plan
        
        self.logger.info(f"Workflow planned: {len(workflow_plan.tasks)} tasks, "
                        f"strategy: {execution_strategy.value}, "
                        f"estimated time: {workflow_plan.estimated_time:.1f}s")
        
        return workflow_plan
    
    def _create_analysis_tasks(self, classification_result: UnifiedFITSResult) -> List[OrchestratorTask]:
        """Create tasks for pure analysis requests"""
        tasks = []
        
        for i, analysis_type in enumerate(classification_result.analysis_types):
            task = OrchestratorTask(
                task_id=f"analysis_{i}_{analysis_type}",
                task_type="analysis",
                agent="fits_service",
                parameters={
                    "analysis_type": analysis_type,
                    "parameters": classification_result.parameters.get(analysis_type, {})
                }
            )
            
            # Add dependencies for sequential analysis (PSD before fitting)
            if analysis_type.startswith("fitting") and i > 0:
                # Fitting tasks depend on PSD if present
                psd_tasks = [t.task_id for t in tasks if "psd" in t.task_id]
                if psd_tasks:
                    task.dependencies = psd_tasks
            
            tasks.append(task)
        
        return tasks
    
    def _create_general_tasks(self, classification_result: UnifiedFITSResult) -> List[OrchestratorTask]:
        """Create tasks for general questions"""
        return [
            OrchestratorTask(
                task_id="astrosage_question",
                task_type="astrosage", 
                agent="astrosage_client",
                parameters={
                    "question_category": classification_result.question_category,
                    "complexity_level": classification_result.complexity_level
                }
            )
        ]
    
    def _create_mixed_tasks(self, classification_result: UnifiedFITSResult) -> List[OrchestratorTask]:
        """Create tasks for mixed requests"""
        tasks = []
        
        # AstroSage task
        astrosage_task = OrchestratorTask(
            task_id="astrosage_mixed",
            task_type="astrosage",
            agent="astrosage_client", 
            parameters={
                "question_category": classification_result.question_category,
                "complexity_level": classification_result.complexity_level,
                "context": "mixed_request"
            }
        )
        
        # Analysis tasks
        analysis_tasks = []
        for i, analysis_type in enumerate(classification_result.analysis_types):
            task = OrchestratorTask(
                task_id=f"analysis_mixed_{i}_{analysis_type}",
                task_type="analysis",
                agent="fits_service",
                parameters={
                    "analysis_type": analysis_type,
                    "parameters": classification_result.parameters.get(analysis_type, {})
                }
            )
            analysis_tasks.append(task)
        
        # Set dependencies based on question context
        if classification_result.question_context == "before_analysis":
            # Question first, then analysis
            for task in analysis_tasks:
                task.dependencies = [astrosage_task.task_id]
            tasks = [astrosage_task] + analysis_tasks
            
        elif classification_result.question_context == "after_analysis":
            # Analysis first, then question with context
            astrosage_task.dependencies = [task.task_id for task in analysis_tasks]
            astrosage_task.parameters["analysis_context"] = True
            tasks = analysis_tasks + [astrosage_task]
            
        else:  # parallel
            # No dependencies - can run in parallel
            tasks = [astrosage_task] + analysis_tasks
        
        return tasks
    
    def _estimate_workflow_time(self, tasks: List[OrchestratorTask]) -> float:
        """Estimate total workflow execution time"""
        
        # Base time estimates per task type
        time_estimates = {
            "analysis": 2.0,      # Average analysis time
            "astrosage": 3.0,     # Average LLM response time
            "classification": 1.5  # Classification time
        }
        
        # Calculate based on dependencies (sequential vs parallel)
        total_time = 0.0
        for task in tasks:
            task_time = time_estimates.get(task.task_type, 2.0)
            if task.dependencies:
                # Sequential - add to total
                total_time += task_time
            else:
                # Parallel - take max
                total_time = max(total_time, task_time)
        
        return total_time
    
    def _estimate_workflow_cost(self, tasks: List[OrchestratorTask]) -> float:
        """Estimate total workflow cost"""
        
        cost_estimates = {
            "analysis": 0.001,    # Compute cost
            "astrosage": 0.003,   # LLM API cost
            "classification": 0.002  # Classification cost
        }
        
        return sum(cost_estimates.get(task.task_type, 0.001) for task in tasks)
    
    async def _execute_workflow(self, 
                              workflow_plan: WorkflowPlan,
                              user_input: str,
                              session_context: SessionContext) -> Dict[str, Any]:
        """Execute the planned workflow"""
        
        # workflow_plan.status = "executing"
        # workflow_plan.start_time = datetime.now()
        
        # execution_results = {}
        
        # try:
        #     if workflow_plan.execution_strategy == ExecutionStrategy.PARALLEL:
        #         # Execute tasks in parallel where possible
        #         execution_results = await self._execute_parallel_workflow(
        #             workflow_plan.tasks, user_input, session_context
        #         )
        #     else:
        #         # Execute tasks sequentially
        #         execution_results = await self._execute_sequential_workflow(
        #             workflow_plan.tasks, user_input, session_context
        #         )
            
        #     workflow_plan.status = "completed"
        #     self.logger.info(f"Workflow {workflow_plan.workflow_id} completed successfully")
            
        # except Exception as e:
        #     workflow_plan.status = "failed"
        #     self.logger.error(f"Workflow {workflow_plan.workflow_id} failed: {str(e)}")
        #     raise
        
        # finally:
        #     workflow_plan.end_time = datetime.now()
        
        # return execution_results
        try:
            return await self._execute_sequential_workflow(
                workflow_plan.tasks, user_input, session_context
            )
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {str(e)}")
            
            # Attempt graceful degradation
            if workflow_plan.tasks:
                partial_results = {}
                for task in workflow_plan.tasks:
                    try:
                        if task.task_type == "analysis":
                            # Try with simplified parameters
                            result = await self._execute_analysis_task_safe(task, session_context)
                            partial_results[task.task_id] = result
                    except Exception as task_error:
                        self.logger.warning(f"Task {task.task_id} failed: {str(task_error)}")
                        continue
                
                return partial_results
            
            raise

    async def _execute_sequential_workflow(self, 
                                         tasks: List[OrchestratorTask],
                                         user_input: str,
                                         session_context: SessionContext) -> Dict[str, Any]:
        """Execute tasks sequentially respecting dependencies"""
        
        results = {}
        completed_tasks = set()
        
        # Build dependency graph
        pending_tasks = tasks.copy()
        
        while pending_tasks:
            # Find tasks with no unmet dependencies
            ready_tasks = []
            for task in pending_tasks:
                if all(dep in completed_tasks for dep in task.dependencies):
                    ready_tasks.append(task)
            
            if not ready_tasks:
                raise Exception("Circular dependency detected in workflow")
            
            # Execute ready tasks
            for task in ready_tasks:
                task.status = "running"
                task.start_time = datetime.now()
                
                try:
                    # Execute task based on type
                    if task.task_type == "analysis":
                        result = await self._execute_analysis_task(task, session_context)
                    elif task.task_type == "astrosage":
                        result = await self._execute_astrosage_task(task, user_input, results, session_context)  # ✅ FIXED
                    else:
                        raise ValueError(f"Unknown task type: {task.task_type}")
                    
                    task.result = result
                    task.status = "completed"
                    results[task.task_id] = result
                    completed_tasks.add(task.task_id)
                    
                    self.logger.debug(f"Task {task.task_id} completed successfully")
                    
                except Exception as e:
                    task.status = "failed"
                    task.error = str(e)
                    self.logger.error(f"Task {task.task_id} failed: {str(e)}")
                    raise
                
                finally:
                    task.end_time = datetime.now()
                
                pending_tasks.remove(task)
        
        return results

    async def _execute_parallel_workflow(self, 
                                       tasks: List[OrchestratorTask],
                                       user_input: str,
                                       session_context: SessionContext) -> Dict[str, Any]:
        """Execute independent tasks in parallel"""
        
        async def execute_single_task(task):
            task.status = "running"
            task.start_time = datetime.now()
            
            try:
                if task.task_type == "analysis":
                    result = await self._execute_analysis_task(task, session_context)
                elif task.task_type == "astrosage":
                    result = await self._execute_astrosage_task(task, user_input, {}, session_context)  # ✅ FIXED
                else:
                    raise ValueError(f"Unknown task type: {task.task_type}")
                
                task.result = result
                task.status = "completed"
                return task.task_id, result
                
            except Exception as e:
                task.status = "failed"
                task.error = str(e)
                raise
            finally:
                task.end_time = datetime.now()
        
        # Execute all tasks in parallel
        task_results = await asyncio.gather(
            *[execute_single_task(task) for task in tasks],
            return_exceptions=True
        )
        
        results = {}
        for task_result in task_results:
            if isinstance(task_result, Exception):
                raise task_result
            task_id, result = task_result
            results[task_id] = result
        
        return results
    
    async def _execute_analysis_task(self, 
                                   task: OrchestratorTask, 
                                   session_context: SessionContext) -> Dict[str, Any]:
        """Execute FITS analysis task"""
        
        analysis_type = task.parameters["analysis_type"]
        parameters = task.parameters["parameters"]
        
        self.logger.info(f"Executing {analysis_type} analysis with parameters: {parameters}")
        
        # Mock implementation - replace with actual FITS service calls
        if analysis_type == "statistics":
            result = {
                "analysis_type": "statistics",
                "results": {
                    "mean": 1.234,
                    "std": 0.567,
                    "median": 1.200,
                    "min": 0.001,
                    "max": 5.678
                },
                "parameters_used": parameters,
                "success": True,
                "processing_time": 1.5
            }
            
        elif analysis_type == "psd":
            result = {
                "analysis_type": "psd",
                "results": {
                    "frequencies": [1e-5, 1e-4, 1e-3],  # Mock data
                    "psd_values": [100.0, 50.0, 25.0],
                    "plot_url": "/plots/psd_12345.png"
                },
                "parameters_used": parameters,
                "success": True,
                "processing_time": 2.0
            }
            
        elif analysis_type.startswith("fitting"):
            result = {
                "analysis_type": analysis_type,
                "results": {
                    "fitted_parameters": {"A": 1.23, "b": 1.45, "n": 0.01},
                    "goodness_of_fit": {"r_squared": 0.95, "chi_squared": 1.23},
                    "plot_url": f"/plots/{analysis_type}_67890.png"
                },
                "parameters_used": parameters,
                "success": True,
                "processing_time": 3.0
            }
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        return result

    def _validate_request(self, user_input: str, context: Dict[str, Any]) -> List[str]:
        """Validate request before processing"""
        issues = []
        
        if not user_input or user_input.strip() == "":
            issues.append("Empty request")
        
        if len(user_input) > 2000:  # Reasonable limit
            issues.append("Request too long")
        
        # Check for analysis requests without files
        analysis_keywords = ["calculate", "compute", "fit", "analyze", "statistics", "psd"]
        if any(keyword in user_input.lower() for keyword in analysis_keywords):
            if not context.get("has_uploaded_files", False):
                issues.append("Analysis requested but no files available")
        
        return issues

    # async def process_request(self, ...):
    #     # Add validation step
    #     validation_issues = self._validate_request(user_input, context or {})
    #     if validation_issues:
    #         return await self._create_validation_error_response(
    #             request_id, validation_issues, start_time
    #         )

    async def _execute_astrosage_task(self, 
                                    task: OrchestratorTask,
                                    user_input: str,
                                    context_results: Dict[str, Any],
                                    session_context: SessionContext) -> Dict[str, Any]:  # ✅ FIXED
        """Execute AstroSage task using real client"""
        
        self.logger.info(f"Executing AstroSage task: {task.parameters}")
        
        try:
            return await self.astrosage_helper.execute_astrosage_task(
                task_params=task.parameters,
                user_input=user_input,
                context_results=context_results,
                conversation_history=getattr(session_context, 'conversation_history', []),
                session_context=session_context.__dict__ if session_context else None
            )
        except Exception as e:
            self.logger.error(f"AstroSage task failed: {str(e)}")
            # Return fallback response
            return {
                "response": f"I apologize, but I'm having trouble connecting to the astronomy knowledge base. However, I can tell you that your question about '{user_input[:50]}...' is interesting!",
                "confidence": 0.3,
                "sources": [],
                "key_concepts": [],
                "follow_up_suggestions": [],
                "processing_time": 0.1,
                "tokens_used": 0,
                "cost": 0.0,
                "success": False,
                "error": str(e),
                "model_used": "fallback"
            }


    
    async def _assemble_response(self, 
                               request_id: str,
                               classification_result: UnifiedFITSResult,
                               execution_results: Dict[str, Any],
                               workflow_plan: WorkflowPlan,
                               session_context: SessionContext,
                               start_time: datetime) -> OrchestratorResponse:
        """Assemble final orchestrated response"""
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Extract analysis results
        analysis_results = {}
        astrosage_response = None
        
        for task_id, result in execution_results.items():
            if "analysis" in task_id:
                analysis_type = result.get("analysis_type", "unknown")
                analysis_results[analysis_type] = result
            elif "astrosage" in task_id:
                astrosage_response = result.get("response")
        
        # Calculate actual cost
        actual_cost = classification_result.cost_estimate + sum(
            result.get("cost", 0.001) for result in execution_results.values()
        )
        
        # Generate explanation for mixed requests
        explanation = None
        if classification_result.is_mixed_request:
            explanation = self._generate_mixed_explanation(
                classification_result, analysis_results, astrosage_response
            )
        
        # Suggest next steps
        suggested_next_steps = self._generate_next_steps(
            classification_result, analysis_results, session_context
        )
        
        # Generate warnings
        warnings = []
        if actual_cost > 0.01:
            warnings.append(f"High processing cost: ${actual_cost:.4f}")
        if processing_time > 10:
            warnings.append(f"Long processing time: {processing_time:.1f}s")
        
        response = OrchestratorResponse(
            request_id=request_id,
            session_id=session_context.session_id,
            success=True,
            primary_intent=classification_result.primary_intent,
            analysis_types=classification_result.analysis_types,
            confidence=classification_result.confidence,
            analysis_results=analysis_results,
            astrosage_response=astrosage_response,
            workflow=workflow_plan,
            processing_time=processing_time,
            total_cost=actual_cost,
            explanation=explanation,
            suggested_next_steps=suggested_next_steps,
            warnings=warnings,
            session_context=session_context
        )

        # Validate response completeness
        if response.primary_intent == "analysis" and not response.analysis_results:
            response.warnings.append("Analysis requested but no results generated")
        
        if response.total_cost > 0.05:  # High cost warning
            response.warnings.append(f"High processing cost: ${response.total_cost:.4f}")
        
        return response
    
    def _generate_mixed_explanation(self, 
                                  classification_result: UnifiedFITSResult,
                                  analysis_results: Dict[str, Any],
                                  astrosage_response: Optional[str]) -> str:
        """Generate explanation for mixed requests"""
        
        explanation_parts = []
        
        if astrosage_response:
            explanation_parts.append(f"**Explanation:** {astrosage_response}")
        
        if analysis_results:
            explanation_parts.append("**Analysis Results:**")
            for analysis_type, result in analysis_results.items():
                explanation_parts.append(f"- {analysis_type}: {result.get('summary', 'Completed successfully')}")
        
        return "\n\n".join(explanation_parts) if explanation_parts else None
    
    def _generate_next_steps(self, 
                           classification_result: UnifiedFITSResult,
                           analysis_results: Dict[str, Any],
                           session_context: SessionContext) -> List[str]:
        """Generate suggested next steps"""
        
        next_steps = []
        
        # Based on what was done, suggest logical next steps
        completed_analyses = set(analysis_results.keys())
        
        if "statistics" in completed_analyses and "psd" not in completed_analyses:
            next_steps.append("Consider computing Power Spectral Density (PSD) for frequency analysis")
        
        if "psd" in completed_analyses and not any("fitting" in a for a in completed_analyses):
            next_steps.append("Try fitting power law models to characterize the noise properties")
        
        if "fitting_power_law" in completed_analyses and "fitting_bending_power_law" not in completed_analyses:
            next_steps.append("Compare with bending power law model for better fits")
        
        # Expertise-based suggestions
        if session_context.user_expertise == "beginner":
            next_steps.append("Ask questions about interpreting your results")
        elif session_context.user_expertise == "advanced":
            next_steps.append("Consider parameter optimization or comparative analysis")
        
        return next_steps[:3]  # Limit to 3 suggestions

    async def _update_session_state(self, 
                                  session_context: SessionContext,
                                  classification_result: UnifiedFITSResult,
                                  response: OrchestratorResponse):
        """Update session state with results"""
        
        if response.analysis_results:
            for analysis_type, result in response.analysis_results.items():
                session_context.analysis_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "analysis_type": analysis_type,
                    "parameters": result.get("parameters_used", {}),
                    "success": result.get("success", True),
                    "processing_time": result.get("processing_time", 0.0)
                })
        
        # Add to parameter history
        if classification_result.parameters:
            for analysis_type, params in classification_result.parameters.items():
                session_context.parameter_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "analysis_type": analysis_type,
                    "parameters": params,
                    "source": classification_result.parameter_source.get(analysis_type, "unknown"),
                    "confidence": classification_result.parameter_confidence.get(analysis_type, 0.0)
                })
        
        # Update user preferences based on usage patterns
        await self._update_user_preferences(session_context, classification_result)
        
        # Update session totals
        session_context.total_cost += response.total_cost
        
        # Update global statistics
        self.stats["total_cost"] += response.total_cost
        total_requests = self.stats["total_requests"]
        self.stats["avg_processing_time"] = (
            (self.stats["avg_processing_time"] * (total_requests - 1) + response.processing_time) / total_requests
        )
    
    async def _update_user_preferences(self, 
                                     session_context: SessionContext,
                                     classification_result: UnifiedFITSResult):
        """Learn and update user preferences from usage patterns"""
        
        # Track parameter preferences
        for analysis_type, params in classification_result.parameters.items():
            pref_key = f"preferred_{analysis_type}_params"
            
            if pref_key not in session_context.user_preferences:
                session_context.user_preferences[pref_key] = {}
            
            # Update preferences for frequently used parameters
            for param_name, param_value in params.items():
                if param_name in ["bins", "low_freq", "high_freq"]:  # Track key parameters
                    session_context.user_preferences[pref_key][param_name] = param_value
        
        # Track analysis type preferences
        if classification_result.analysis_types:
            pref_key = "frequent_analysis_types"
            if pref_key not in session_context.user_preferences:
                session_context.user_preferences[pref_key] = {}
            
            for analysis_type in classification_result.analysis_types:
                current_count = session_context.user_preferences[pref_key].get(analysis_type, 0)
                session_context.user_preferences[pref_key][analysis_type] = current_count + 1
    
    async def _cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session_context in self.active_sessions.items():
            if current_time - session_context.last_activity > self.session_ttl:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
            self.logger.debug(f"Cleaned up expired session: {session_id}")
    
    async def _create_error_response(self, 
                                   request_id: str,
                                   session_id: str,
                                   error_message: str,
                                   start_time: datetime) -> OrchestratorResponse:
        """Create error response"""
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return OrchestratorResponse(
            request_id=request_id,
            session_id=session_id,
            success=False,
            primary_intent="error",
            analysis_types=[],
            confidence=0.0,
            processing_time=processing_time,
            total_cost=0.0,
            warnings=[f"Processing failed: {error_message}"],
            suggested_next_steps=[
                "Check your input format",
                "Try a simpler request",
                "Contact support if the issue persists"
            ]
        )
    
    # ========================================
    # SESSION MANAGEMENT METHODS
    # ========================================
    
    async def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive session information"""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        
        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "conversation_count": session.conversation_count,
            "total_cost": session.total_cost,
            "user_expertise": session.user_expertise,
            "uploaded_files": session.uploaded_files,
            "analysis_history_count": len(session.analysis_history),
            "parameter_history_count": len(session.parameter_history),
            "user_preferences": session.user_preferences
        }
    
    async def update_session_context(self, 
                                   session_id: str,
                                   updates: Dict[str, Any]) -> bool:
        """Update session context"""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        
        if "user_expertise" in updates:
            session.user_expertise = updates["user_expertise"]
        
        if "uploaded_files" in updates:
            session.uploaded_files = updates["uploaded_files"]
        
        if "user_preferences" in updates:
            session.user_preferences.update(updates["user_preferences"])
        
        session.last_activity = datetime.now()
        return True
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            self.logger.info(f"Deleted session: {session_id}")
            return True
        return False
    
    # ========================================
    # STATISTICS AND MONITORING
    # ========================================
    
    def get_orchestrator_stats(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator statistics"""
        return {
            "orchestrator_id": self.orchestrator_id,
            "uptime": (datetime.now() - datetime.now()).total_seconds(),  # Would track actual uptime
            "requests": {
                "total": self.stats["total_requests"],
                "successful": self.stats["successful_requests"],
                "failed": self.stats["failed_requests"],
                "success_rate": (
                    self.stats["successful_requests"] / max(self.stats["total_requests"], 1)
                )
            },
            "sessions": {
                "total_created": self.stats["total_sessions"],
                "currently_active": len(self.active_sessions),
                "average_conversations_per_session": (
                    sum(s.conversation_count for s in self.active_sessions.values()) / 
                    max(len(self.active_sessions), 1)
                )
            },
            "performance": {
                "avg_processing_time": self.stats["avg_processing_time"],
                "total_cost": self.stats["total_cost"],
                "avg_cost_per_request": (
                    self.stats["total_cost"] / max(self.stats["total_requests"], 1)
                )
            },
            "request_distribution": self.stats["request_types"],
            "active_workflows": len(self.active_workflows)
        }
    
    def get_session_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all active sessions"""
        return [
            {
                "session_id": session_id,
                "user_id": session.user_id,
                "conversation_count": session.conversation_count,
                "total_cost": session.total_cost,
                "last_activity": session.last_activity.isoformat(),
                "user_expertise": session.user_expertise
            }
            for session_id, session in self.active_sessions.items()
        ]
    
    # ========================================
    # UTILITY METHODS
    # ========================================
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of orchestrator and dependencies"""
        health_status = {
            "orchestrator": "healthy",
            "classification_agent": "unknown",
            "fits_service": "unknown", 
            "astrosage_client": "unknown",
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Test classification agent
            test_result = await self.classification_agent.process_request(
                "test", {"has_uploaded_files": False}
            )
            health_status["classification_agent"] = "healthy"
        except Exception as e:
            health_status["classification_agent"] = f"unhealthy: {str(e)}"
        
        # TODO: Test other services when implemented
        # health_status["fits_service"] = await self.fits_service.health_check()
        # health_status["astrosage_client"] = await self.astrosage_client.health_check()
        
        return health_status
    
    async def shutdown(self):
        """Graceful shutdown of orchestrator"""
        self.logger.info("Shutting down orchestrator...")
        
        # Wait for active workflows to complete
        if self.active_workflows:
            self.logger.info(f"Waiting for {len(self.active_workflows)} workflows to complete...")
            # In production, would wait for workflows with timeout
            await asyncio.sleep(1)
        
        # Clear sessions and workflows
        self.active_sessions.clear()
        self.active_workflows.clear()
        
        self.logger.info("Orchestrator shutdown complete")


# ========================================
# FACTORY FUNCTIONS AND UTILITIES
# ========================================

def create_fits_orchestrator() -> FITSAnalysisOrchestrator:
    """Factory function to create FITS orchestrator"""
    return FITSAnalysisOrchestrator()


async def demo_orchestrator():
    """Demo function to test orchestrator functionality"""
    print("🎭 FITS Analysis Orchestrator Demo")
    print("=" * 60)
    
    orchestrator = create_fits_orchestrator()
    
    # Test requests
    test_requests = [
        {
            "input": "Calculate mean and standard deviation for my data",
            "context": {"uploaded_files": ["neutron_star_data.fits"], "user_expertise": "intermediate"}
        },
        {
            "input": "What is a power spectral density?",
            "context": {"user_expertise": "beginner"}
        },
        {
            "input": "What is PSD? Then compute it for my data with 4000 bins",
            "context": {"uploaded_files": ["pulsar_data.fits"], "user_expertise": "intermediate"}
        },
        {
            "input": "Compute PSD and fit both power law models, then explain what the break frequency means",
            "context": {"uploaded_files": ["xray_data.fits"], "user_expertise": "advanced"}
        }
    ]
    
    session_id = None  # Let orchestrator create session
    
    for i, test_request in enumerate(test_requests, 1):
        print(f"\n{'='*80}")
        print(f"Test Request {i}: {test_request['input']}")
        print(f"Context: {test_request['context']}")
        print(f"{'='*80}")
        
        try:
            # Process request
            response = await orchestrator.process_request(
                user_input=test_request["input"],
                session_id=session_id,
                user_id="demo_user",
                context=test_request["context"]
            )
            
            # Use same session for subsequent requests
            if session_id is None:
                session_id = response.session_id
            
            # Display results
            print(f"✅ Success: {response.success}")
            print(f"🎯 Intent: {response.primary_intent}")
            print(f"📊 Analysis Types: {response.analysis_types}")
            print(f"🎭 Confidence: {response.confidence:.2f}")
            print(f"⏱️  Processing Time: {response.processing_time:.3f}s")
            print(f"💰 Cost: ${response.total_cost:.6f}")
            
            if response.analysis_results:
                print(f"\n📈 Analysis Results:")
                for analysis_type, result in response.analysis_results.items():
                    print(f"   {analysis_type}: {result.get('success', 'Unknown status')}")
            
            if response.astrosage_response:
                print(f"\n🔬 AstroSage Response:")
                # print(f"   {response.astrosage_response[:100]}...")
                print(f"   {response.astrosage_response}")
            
            if response.explanation:
                print(f"\n💡 Explanation:")
                print(f"   {response.explanation[:150]}...")
            
            if response.suggested_next_steps:
                print(f"\n➡️  Suggested Next Steps:")
                for step in response.suggested_next_steps:
                    print(f"   • {step}")
            
            if response.warnings:
                print(f"\n⚠️  Warnings:")
                for warning in response.warnings:
                    print(f"   • {warning}")
            
            # Show workflow info
            if response.workflow:
                workflow = response.workflow
                print(f"\n🔄 Workflow Info:")
                print(f"   Strategy: {workflow.execution_strategy.value}")
                print(f"   Tasks: {len(workflow.tasks)}")
                print(f"   Status: {workflow.status}")
                print(f"   Estimated Time: {workflow.estimated_time:.1f}s")
                print(f"   Estimated Cost: ${workflow.estimated_cost:.6f}")
        
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        await asyncio.sleep(0.5)
    
    # Show final statistics
    print(f"\n{'='*80}")
    print("📊 ORCHESTRATOR STATISTICS")
    print(f"{'='*80}")
    
    stats = orchestrator.get_orchestrator_stats()
    for category, data in stats.items():
        print(f"\n{category.upper()}:")
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.3f}")
                else:
                    print(f"   {key}: {value}")
        else:
            print(f"   {data}")
    
    # Show session info
    if session_id:
        session_info = await orchestrator.get_session_info(session_id)
        print(f"\n📋 SESSION INFO:")
        for key, value in session_info.items():
            print(f"   {key}: {value}")
    
    # Health check
    health = await orchestrator.health_check()
    print(f"\n🏥 HEALTH CHECK:")
    for service, status in health.items():
        print(f"   {service}: {status}")
    
    print(f"\n🎉 Orchestrator demo completed!")


# Main execution
if __name__ == "__main__":
    async def main():
        """Main demo function"""
        try:
            await demo_orchestrator()
        except KeyboardInterrupt:
            print("\n👋 Demo interrupted")
        except Exception as e:
            print(f"\n❌ Demo failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Run the demo
    asyncio.run(main())