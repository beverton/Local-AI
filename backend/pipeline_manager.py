"""
Pipeline Manager - Orchestriert Agent-Workflows
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineManager:
    """Verwaltet und orchestriert Agent-Pipelines"""
    
    def __init__(self, agent_manager):
        self.agent_manager = agent_manager
        self.pipelines: Dict[str, Dict[str, Any]] = {}
        self._set_managers_func = None
    
    def create_pipeline(self, conversation_id: str, pipeline_name: str, 
                       steps: List[Dict[str, Any]]) -> str:
        """
        Erstellt eine Pipeline
        
        Args:
            conversation_id: Die ID der Conversation
            pipeline_name: Name der Pipeline
            steps: Liste von Pipeline-Schritten
                Jeder Schritt: {"agent_type": "...", "model_id": "...", "input": "..."}
        
        Returns:
            Pipeline-ID
        """
        import uuid
        pipeline_id = str(uuid.uuid4())
        
        self.pipelines[pipeline_id] = {
            "id": pipeline_id,
            "conversation_id": conversation_id,
            "name": pipeline_name,
            "steps": steps,
            "created_at": datetime.now().isoformat(),
            "status": "pending"
        }
        
        logger.info(f"Pipeline {pipeline_id} erstellt: {pipeline_name}")
        return pipeline_id
    
    def execute_pipeline(self, pipeline_id: str, initial_input: str) -> Dict[str, Any]:
        """
        Führt eine Pipeline aus
        
        Args:
            pipeline_id: Die ID der Pipeline
            initial_input: Der initiale Input (z.B. Text für Prompt-Agent)
            
        Returns:
            Dict mit Ergebnissen jedes Schritts
        """
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} nicht gefunden")
        
        pipeline = self.pipelines[pipeline_id]
        conversation_id = pipeline["conversation_id"]
        steps = pipeline["steps"]
        
        pipeline["status"] = "running"
        results = []
        current_input = initial_input
        
        try:
            for i, step in enumerate(steps):
                step_result = {
                    "step": i + 1,
                    "agent_type": step.get("agent_type"),
                    "input": current_input[:100] + "..." if len(current_input) > 100 else current_input,
                    "output": None,
                    "error": None
                }
                
                try:
                    # Erstelle oder finde Agent
                    agent_id = step.get("agent_id")
                    if not agent_id:
                        # Erstelle neuen Agent
                        agent_id = self.agent_manager.create_agent(
                            conversation_id=conversation_id,
                            agent_type=step["agent_type"],
                            model_id=step.get("model_id"),
                            set_managers_func=self._set_managers_func if self._set_managers_func else None
                        )
                    
                    agent = self.agent_manager.get_agent(conversation_id, agent_id)
                    if not agent:
                        raise ValueError(f"Agent {agent_id} nicht gefunden")
                    
                    # Führe Schritt aus
                    logger.info(f"Pipeline Schritt {i+1}/{len(steps)}: {step['agent_type']}")
                    output = agent.process_message(current_input)
                    step_result["output"] = output
                    current_input = output  # Output wird Input für nächsten Schritt
                    
                except Exception as e:
                    logger.error(f"Fehler in Pipeline-Schritt {i+1}: {e}")
                    step_result["error"] = str(e)
                    pipeline["status"] = "error"
                    break
                
                results.append(step_result)
            
            if pipeline["status"] != "error":
                pipeline["status"] = "completed"
            
            pipeline["results"] = results
            pipeline["final_output"] = current_input
            
            logger.info(f"Pipeline {pipeline_id} abgeschlossen: {pipeline['status']}")
            return {
                "pipeline_id": pipeline_id,
                "status": pipeline["status"],
                "results": results,
                "final_output": current_input
            }
            
        except Exception as e:
            pipeline["status"] = "error"
            logger.error(f"Pipeline-Fehler: {e}")
            raise
    
    def set_managers_func(self, func):
        """Setzt die Funktion zum Setzen der Manager"""
        self._set_managers_func = func
    
    def get_pipeline(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Gibt eine Pipeline zurück"""
        return self.pipelines.get(pipeline_id)
    
    def get_conversation_pipelines(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Gibt alle Pipelines einer Conversation zurück"""
        return [
            pipeline for pipeline in self.pipelines.values()
            if pipeline["conversation_id"] == conversation_id
        ]

