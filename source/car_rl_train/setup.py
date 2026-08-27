"""car_rl_train Isaac Lab 外部扩展的安装元数据。"""

from setuptools import find_packages, setup

setup(
    name="car_rl_train",
    version="0.1.0",
    description="基于 Isaac Lab 的移动机器人局部导航控制器训练工程",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["numpy", "PyYAML", "psutil"],
    extras_require={"export": ["onnx>=1.16", "onnxruntime>=1.18"]},
    python_requires=">=3.11",
    zip_safe=False,
)
