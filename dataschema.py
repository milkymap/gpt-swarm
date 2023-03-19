from enum import Enum
from typing import List, Dict, Optional, Any, Tuple

from pydantic import BaseModel

class REQUEST_TYPE(str, Enum):
        TIME2SLEEP:str='time2sleep' 
        AVAILABLE_TOKEN:str='available_token'


class Role(str, Enum):
    USER:str='user'
    SYSTEM:str='system'
    ASSISTANT:str='assistant'

class Message(BaseModel):
    role:Role 
    content:str 