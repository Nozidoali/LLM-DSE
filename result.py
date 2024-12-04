from enum import Enum
from typing import Dict, List, Optional, Union

class DesignParameter:
    def __init__(self, name: str = ''):
        self.name: str = name
        self.default: Union[str, int] = 1
        self.option_expr: str = ''
        self.scope: List[str] = []
        self.order: Dict[str, str] = {}
        self.deps: List[str] = []
        self.child: List[str] = []
        self.value: Union[str, int] = 1

DesignSpace = Dict[str, DesignParameter]
DesignPoint = Dict[str, Union[int, str]]

class Result:
    class RetCode(Enum):
        PASS = 0
        UNAVAILABLE = -1
        ANALYZE_ERROR = -2
        EARLY_REJECT = -3
        TIMEOUT = -4
        DUPLICATED = -5

    def __init__(self, ret_code_str: str = 'PASS'):
        self.point: Optional[DesignPoint] = None
        self.ret_code: Result.RetCode = self.RetCode[ret_code_str]
        self.valid: bool = False
        self.path: Optional[str] = None
        self.quality: float = -float('inf')
        self.perf: float = 0.0
        self.res_util: Dict[str, float] = {
            'util-BRAM': 0, 'util-DSP': 0, 'util-LUT': 0, 'util-FF': 0,
            'total-BRAM': 0, 'total-DSP': 0, 'total-LUT': 0, 'total-FF': 0
        }
        self.eval_time: float = 0.0

class MerlinResult(Result):
    def __init__(self, ret_code_str: str = 'PASS'):
        super().__init__(ret_code_str)
        self.criticals: List[str] = []
        self.code_hash: Optional[str] = None
