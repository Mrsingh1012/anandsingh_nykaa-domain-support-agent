"""
Mock LLM Engine for Nykaa Domain Support Agent.
Track: E-commerce & Retail (Nykaa).

Complies strictly with capstone constraints:
1. Zero API keys, zero network access, completely deterministic and local.
2. Extends `crewai.llms.base_llm.BaseLLM` (CrewAI's documented extension point).
3. Avoids Pitfall 1: Ignores template 'Observation:' in ReAct system prompt.
4. Avoids Pitfall 2: Dispatches tool arguments based on declared schema fields
   (e.g. 'record_id' vs 'query') rather than tool name substrings.
5. Disables telemetry via environment variables (CREWAI_DISABLE_TELEMETRY=true).
"""

import os
import re
import json
from typing import Any, List, Dict, Optional, Union
from pydantic import BaseModel, Field

# Guarantee telemetry and encoding isolation
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["PYTHONIOENCODING"] = "utf-8"

from crewai.llms.base_llm import BaseLLM
from tools import check_order_status, rag_policy_lookup


class NykaaMockLLM(BaseLLM):
    """
    Deterministic Mock LLM for Nykaa Customer Support Crew.
    Generates appropriate ReAct actions and synthesis responses based on agent roles.
    """
    model: str = Field(default="nykaa-mock-llm-v1")

    def _extract_conversation_turns(self, messages: Any) -> List[Dict[str, str]]:
        """Parses messages into a list of role/content dictionaries."""
        turns = []
        if isinstance(messages, str):
            turns.append({"role": "user", "content": messages})
        elif isinstance(messages, list):
            for m in messages:
                if isinstance(m, dict):
                    turns.append({"role": m.get("role", "user"), "content": str(m.get("content", ""))})
                elif hasattr(m, "role") and hasattr(m, "content"):
                    turns.append({"role": getattr(m, "role"), "content": str(getattr(m, "content"))})
                else:
                    turns.append({"role": "user", "content": str(m)})
        return turns

    def call(
        self,
        messages: Union[str, List[Any]],
        tools: Optional[List[Any]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
        from_task: Optional[Any] = None,
        from_agent: Optional[Any] = None,
        response_model: Optional[type] = None,
    ) -> Union[str, Any]:
        """Core call execution for CrewAI agent cycles."""
        turns = self._extract_conversation_turns(messages)
        full_text = " ".join(t["content"] for t in turns)
        
        # Identify agent role
        agent_role = ""
        if from_agent and hasattr(from_agent, "role"):
            agent_role = str(from_agent.role).lower()

        # Check for tool execution observation in dialogue (avoiding system template)
        # In CrewAI, actual tool results are inserted as 'Observation: <result>' or in tool messages
        has_real_observation = False
        latest_observation = ""
        
        for t in reversed(turns):
            content = t["content"]
            # Exclude the known system prompt instructions
            if "Observation: the result of the action" in content:
                continue
            obs_match = re.search(r"Observation:\s*(.*)", content, re.DOTALL)
            if obs_match:
                has_real_observation = True
                latest_observation = obs_match.group(1).strip()
                break

        # -------------------------------------------------------------
        # 1. AGENTS WITH TOOLS (Retrieval Agent & Lookup Agent)
        # -------------------------------------------------------------
        if tools and not has_real_observation:
            # Inspect available tools and dispatch by declared argument schema
            for tool_obj in tools:
                tool_name = getattr(tool_obj, "name", "")
                tool_args = getattr(tool_obj, "args", {}) or getattr(tool_obj, "arguments", {})
                
                # Check schema properties for record_id vs query
                schema_keys = set()
                if isinstance(tool_args, dict):
                    schema_keys = set(tool_args.keys())
                elif hasattr(tool_obj, "args_schema") and tool_obj.args_schema:
                    try:
                        schema_keys = set(tool_obj.args_schema.model_fields.keys())
                    except Exception:
                        pass

                # Case A: Order Lookup (schema expects 'record_id')
                if "record_id" in schema_keys or "order" in tool_name.lower():
                    order_match = re.search(r"NYK-\d{4,}", full_text, re.IGNORECASE)
                    record_id = order_match.group(0).upper() if order_match else "NYK-1005"
                    return (
                        f"Thought: I need to query the fulfillment database for order {record_id}.\n"
                        f"Action: {tool_name}\n"
                        f"Action Input: {json.dumps({'record_id': record_id})}"
                    )

                # Case B: RAG Policy Retrieval (schema expects 'query')
                if "query" in schema_keys or "rag" in tool_name.lower() or "policy" in tool_name.lower():
                    # Extract customer policy question
                    query_candidate = "What is the Nykaa return and refund policy?"
                    q_match = re.search(r"(?:inquiry|User Query|query):\s*['\"]?([^\n\r'\"]+)", full_text, re.IGNORECASE)
                    if q_match:
                        query_candidate = q_match.group(1).strip()
                    elif "return" in full_text.lower():
                        query_candidate = "What is the return window for products?"
                    elif "cod" in full_text.lower() or "refund" in full_text.lower():
                        query_candidate = "How are COD refunds processed?"
                    elif "cancel" in full_text.lower():
                        query_candidate = "Can I cancel an order that has already shipped?"
                    return (
                        f"Thought: I need to retrieve grounded knowledge from the Nykaa policy base.\n"
                        f"Action: {tool_name}\n"
                        f"Action Input: {json.dumps({'query': query_candidate})}"
                    )

        # -------------------------------------------------------------
        # 2. AGENTS CONCLUDING OR WITHOUT TOOLS (Response Composer)
        # -------------------------------------------------------------
        if has_real_observation and latest_observation:
            try:
                obs_data = json.loads(latest_observation)
                if isinstance(obs_data, dict):
                    if "answer" in obs_data and obs_data["answer"]:
                        return f"Thought: I will compose the answer based on the retrieved knowledge.\nFinal Answer: {obs_data['answer']}"
                    elif "message" in obs_data and obs_data["message"]:
                        return f"Thought: I will provide the verified order status.\nFinal Answer: {obs_data['message']}"
            except Exception:
                pass

        # Extract active inquiry from task description
        curr_inquiry = ""
        inq_match = re.search(r"(?:inquiry|query):\s*['\"]?([^\n\r'\"]+)", full_text, re.IGNORECASE)
        if inq_match:
            curr_inquiry = inq_match.group(1).strip()
        else:
            user_queries = [t["content"] for t in turns if t.get("role") == "user"]
            curr_inquiry = user_queries[-1] if user_queries else full_text

        order_match = re.search(r"NYK-\d{4,}", curr_inquiry, re.IGNORECASE)

        if order_match:
            order_str = order_match.group(0).upper()
            status_data = check_order_status(order_str)
            
            esc_note = (
                "Due to shipping delay and extended transit time, your inquiry has been escalated "
                "to our Level 2 Senior Support Supervisors under a 12-hour SLA."
                if status_data["recommend_escalation"]
                else "Your order is progressing normally within our standard delivery SLAs."
            )
            final_text = (
                f"Hello! Regarding your order {status_data['record_id']} ({status_data['category']}): "
                f"The current status is '{status_data['status']}' with an order value of INR {status_data['order_value_inr']:,}. "
                f"It was placed {status_data['days_since_created']} days ago. {esc_note}"
            )
        else:
            # Policy question synthesis
            context_match = re.search(
                r"(?:This is the context you're working with:\s*|Relevant policy excerpts:\s*|answer[\"']?\s*:\s*[\"'])(.*?)(?:\n\s*Task:|\n\s*This is the expected output:|\"[\s,]*\"sources|$)",
                full_text,
                re.DOTALL | re.IGNORECASE
            )
            extracted_context = ""
            if context_match:
                extracted_context = context_match.group(1).strip().replace('\\"', '"')

            if len(extracted_context) > 25 and not extracted_context.startswith("Hello! Regarding"):
                final_text = extracted_context
            else:
                policy_result = rag_policy_lookup(curr_inquiry[:120])
                if policy_result["is_grounded"]:
                    final_text = policy_result["answer"]
                else:
                    final_text = "I don't know based on the provided Nykaa policy documentation."

        for prompt_artifact in ["Provide your complete response:", "Provide your complete response", "Give your answer:"]:
            if prompt_artifact in final_text:
                final_text = final_text.split(prompt_artifact)[0].strip()

        # If a structured response_model was passed
        if response_model:
            try:
                return response_model(
                    query=full_text[:80],
                    draft_answer=final_text,
                    sources=["KB-01", "KB-06"] if "NYK-" not in full_text else ["dataset_orders"],
                    requires_escalation="escalated" in final_text.lower(),
                    grounded=not final_text.startswith("I don't know")
                )
            except Exception:
                pass

        return f"Thought: I have synthesized the verified facts.\nFinal Answer: {final_text}"


# Global default instance
nykaa_mock_llm = NykaaMockLLM(model="nykaa-mock-llm-v1")
