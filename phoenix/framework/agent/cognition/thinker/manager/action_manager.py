from typing import Dict, List, Union
from uuid import UUID
from phoenix.framework.agent.cognition.schema import ActionSchema

class ActionManager:
    def __init__(self) -> None:
        self.actions: Dict[str, ActionSchema] = {}

    def _check_action_id(self, action_id: Union[str, UUID]) -> str:
        if isinstance(action_id, UUID):
            action_id = str(action_id)
        if not isinstance(action_id, str):
            raise TypeError("action_id must be a string or UUID")
        if not action_id:
            raise ValueError("action_id cannot be empty")
        if action_id not in self.actions:
            raise ValueError(f"Action with id {action_id} does not exist")
        return action_id

    def push_action(self, action: ActionSchema) -> ActionSchema:
        if not isinstance(action, ActionSchema):
            raise TypeError("action must be an ActionSchema instance")
        self.actions[str(action.action_id)] = action
        return action

    def get_action(self, action_id: Union[str, UUID]) -> ActionSchema:
        action_id_str = self._check_action_id(action_id)
        return self.actions[action_id_str]

    def update_action(self, action: ActionSchema) -> None:
        if not isinstance(action, ActionSchema):
            raise TypeError("action must be an ActionSchema instance")
        action_id_str = self._check_action_id(action.action_id)
        self.actions[action_id_str] = action

    def delete_action(self, action_id: Union[str, UUID]) -> None:
        action_id_str = self._check_action_id(action_id)
        del self.actions[action_id_str]

    def list_actions(self) -> List[ActionSchema]:
        return list(self.actions.values())
    
    def clear_actions(self) -> None:
        self.actions.clear()