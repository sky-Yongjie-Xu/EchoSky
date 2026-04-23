from importlib import import_module

class CardiacEchoPipeline:
    def __init__(self, config):
        self.modules = {}
        self.load_all_modules(config["all_modules"])  # 先加载所有模块

    def load_all_modules(self, module_names):
        """一次性加载所有可用模块，不自动运行"""
        for name in module_names:
            module_class = import_module(f"modules.{name}")
            self.modules[name] = module_class()

    def run_module(self, module_name, echo_data, prev_results=None):
        """
        单独执行某一个功能（最常用！）
        :param module_name: 模块名，如 preprocess/segmentation/quantify
        :param echo_data: 超声图像数据
        :param prev_results: 前面步骤的结果（可选）
        """
        if prev_results is None:
            prev_results = {}

        if module_name not in self.modules:
            raise ValueError(f"模块 {module_name} 未加载！")

        # 只运行你选择的功能
        result = self.modules[module_name].process(echo_data, prev_results)
        return result

    def run_selected(self, module_list, echo_data):
        """
        批量执行你指定的一组功能（自定义顺序）
        不是固定流水线！是你自己选顺序！
        """
        results = {}
        for name in module_list:
            results[name] = self.run_module(name, echo_data, results)
        return results