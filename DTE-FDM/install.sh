pip install --upgrade pip  -i https://pypi.tuna.tsinghua.edu.cn/simple # enable PEP 660 support 
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install -e ".[train]" -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install flash-attn --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple

git pull 
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple

# if you see some import errors when you upgrade,
# please try running the command below (without #)
# pip install flash-attn --no-build-isolation --no-cache-dir