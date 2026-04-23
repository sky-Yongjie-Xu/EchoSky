from core.engine import CardiacEchoEngine

if __name__ == "__main__":
    engine = CardiacEchoEngine()
    engine.list_modules()  # 自动打印所有可用功能

    # engine.run("lv_segmentation", save_video=True)

    engine.run("lv_ef_prediction")
