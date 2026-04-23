class CardiacEchoPipeline:
    def __init__(self, config):
        self.modules = {}
        self.load_modules(config["active_modules"])
    
    def load_modules(self, module_names):
        """动态加载模块"""
        for name in module_names:
            module_class = import_module(f"modules.{name}")
            self.modules[name] = module_class()
    
    def process(self, echo_data):
        """流水线处理"""
        results = {}
        for name, module in self.modules.items():
            results[name] = module.process(echo_data, results)
        return results