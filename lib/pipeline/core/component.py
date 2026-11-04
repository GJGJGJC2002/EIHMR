"""Component - Pipeline 组件抽象基类"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging
import time
import torch

from .data import PipelineData


class Component(ABC):
    """
    Pipeline 组件的抽象基类
    
    所有 Pipeline 组件都必须继承此类并实现抽象方法。
    
    Attributes:
        name: 组件名称
        config: 组件配置
        device: 计算设备
        logger: 日志记录器
    
    Example:
        class MyComponent(Component):
            def setup(self):
                self.model = load_model()
            
            def execute(self, data: PipelineData) -> PipelineData:
                result = self.model(data.images)
                data.result = result
                return data
            
            def validate_input(self, data: PipelineData) -> bool:
                return data.images is not None
    """
    
    # 组件类型标识
    COMPONENT_TYPE: str = "base"
    
    # 默认配置
    DEFAULT_CONFIG: Dict[str, Any] = {}
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        初始化组件
        
        Args:
            name: 组件实例名称
            config: 组件配置字典
        """
        self.name = name
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        
        # 状态
        self._is_setup = False
        self._cache = {}
        
        # 性能统计
        self._execution_times: List[float] = []
    
    @abstractmethod
    def setup(self):
        """
        初始化组件（加载模型、资源等）
        
        子类必须实现此方法。在 Pipeline 执行前会调用一次。
        """
        pass
    
    @abstractmethod
    def execute(self, data: PipelineData) -> PipelineData:
        """
        执行组件功能
        
        子类必须实现此方法。
        
        Args:
            data: Pipeline 数据容器
        
        Returns:
            处理后的 Pipeline 数据容器
        """
        pass
    
    @abstractmethod
    def validate_input(self, data: PipelineData) -> bool:
        """
        验证输入数据是否满足组件要求
        
        子类必须实现此方法。
        
        Args:
            data: Pipeline 数据容器
        
        Returns:
            输入是否有效
        """
        pass
    
    def cleanup(self):
        """
        清理资源（释放模型、GPU 内存等）
        
        子类可以重写此方法来释放资源。
        Pipeline 会在每个组件执行完后自动调用此方法，
        下次需要时通过 ensure_setup() 重新加载。
        """
        self._cache.clear()
        if torch.cuda.is_available():
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        self._is_setup = False
        self.logger.debug(f"Component {self.name} cleaned up")
    
    def ensure_setup(self):
        """确保组件已初始化"""
        if not self._is_setup:
            self.logger.info(f"Setting up {self.name}...")
            start_time = time.time()
            self.setup()
            self._is_setup = True
            elapsed = time.time() - start_time
            self.logger.info(f"Setup completed in {elapsed:.2f}s")
    
    def __call__(self, data: PipelineData) -> PipelineData:
        """
        执行组件（带计时和日志）
        
        Args:
            data: Pipeline 数据容器
        
        Returns:
            处理后的数据容器
        """
        # 确保已初始化
        self.ensure_setup()
        
        # 记录阶段
        data.record_stage(self.name)
        
        # 验证输入
        if not self.validate_input(data):
            raise ValueError(f"Invalid input for component {self.name}")
        
        # 执行并计时
        self.logger.info(f"Executing {self.name}...")
        start_time = time.time()
        
        try:
            result = self.execute(data)
        except Exception as e:
            self.logger.error(f"Error in {self.name}: {str(e)}")
            data.error_message = str(e)
            raise
        
        elapsed = time.time() - start_time
        self._execution_times.append(elapsed)
        self.logger.info(f"Completed in {elapsed:.2f}s")
        
        return result
    
    # === 缓存管理 ===
    
    def cache_get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        return self._cache.get(key)
    
    def cache_set(self, key: str, value: Any):
        """设置缓存值"""
        self._cache[key] = value
    
    def cache_clear(self):
        """清除缓存"""
        self._cache.clear()
    
    # === 性能统计 ===
    
    def get_avg_execution_time(self) -> float:
        """获取平均执行时间"""
        if not self._execution_times:
            return 0.0
        return sum(self._execution_times) / len(self._execution_times)
    
    def get_total_execution_time(self) -> float:
        """获取总执行时间"""
        return sum(self._execution_times)
    
    # === 配置管理 ===
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self.config.get(key, default)
    
    def set_config(self, key: str, value: Any):
        """设置配置值"""
        self.config[key] = value
    
    # === 工具方法 ===
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """将张量移动到组件设备"""
        return tensor.to(self.device)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', device='{self.device}')"


class BackendComponent(Component):
    """
    支持多后端的组件基类
    
    子类可以注册多个后端实现，通过配置选择使用哪个后端。
    
    Example:
        class DetectionComponent(BackendComponent):
            BACKENDS = {
                'vitdet': VitDetBackend,
                'yolo': YoloBackend,
            }
    """
    
    # 后端注册表（子类重写）
    BACKENDS: Dict[str, type] = {}
    
    # 默认后端
    DEFAULT_BACKEND: str = ""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config)
        
        # 获取后端类型
        self.backend_type = self.config.get('backend', self.DEFAULT_BACKEND)
        if not self.backend_type:
            raise ValueError(f"Backend type not specified for {self.name}")
        
        if self.backend_type not in self.BACKENDS:
            raise ValueError(
                f"Unknown backend '{self.backend_type}' for {self.__class__.__name__}. "
                f"Available backends: {list(self.BACKENDS.keys())}"
            )
        
        # 创建后端实例
        backend_cls = self.BACKENDS[self.backend_type]
        self.backend = backend_cls(self.config)
    
    def setup(self):
        """初始化后端"""
        self.backend.setup()
    
    def cleanup(self):
        """清理后端资源"""
        self.backend.cleanup()
        super().cleanup()
    
    @classmethod
    def register_backend(cls, name: str, backend_cls: type):
        """注册新后端"""
        cls.BACKENDS[name] = backend_cls
    
    @classmethod
    def list_backends(cls) -> List[str]:
        """列出可用后端"""
        return list(cls.BACKENDS.keys())


class Backend(ABC):
    """
    后端实现的抽象基类
    
    后端负责具体的模型加载和推理逻辑。
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._is_setup = False
    
    @abstractmethod
    def setup(self):
        """初始化模型"""
        pass
    
    def cleanup(self):
        """清理资源"""
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        self._is_setup = False
    
    def ensure_setup(self):
        """确保已初始化"""
        if not self._is_setup:
            self.setup()
            self._is_setup = True


