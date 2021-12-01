# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def main():
    # 生成数据画图
    x=np.arange(0,14,0.01)
    y=np.sin(x)
    plt.figure()
    plt.plot(x,y)
    # plt.show()

    # 转base64
    figfile = BytesIO()
    plt.savefig(figfile, format='svg')
    figfile.seek(0)
    figdata_data = base64.b64encode(figfile.getvalue()).decode()
    filename='png3.html'
    with open(filename,'w') as f:
        f.write(f'<img src="data:image/svg+xml;base64,{figdata_data}"/>')

if __name__ == '__main__':
    main()
