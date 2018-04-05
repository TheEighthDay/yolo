<h1>经验总结</h1>
<h2>1.概况</h2>
这是一个利用yolo算法实现车流识别的系统。<br>
硬件基础是cpui7-7700HQ，显卡英伟达1050TI。<br>
基于win10，python3。<br>
python的安装库要求深度学习框架tensflow，keras和一些常用机器学习库。<br>
是否选择gpu加速看个人情况，这里使用cuda8.0，cudnn6.0。<br>
<h2>2.yolo简述</h2>
YOLO ("you only look once")是一种流行的算法，因为它实现了高精度的同时还能够在实时运行。
<h3>1.目标的表示。</h3>
<h3>我们所熟知一般的卷积神经网络，经过若干次池化和卷积，在全连接层后利用softmax分类或者sigmoid判断物体是否存在。而对于yolo，它将一个图片分成19×19的网格，在每个网格里寻找物体。物体的表现形式为 (pc,bx,by,bh,bw,c)，pc代表物体存在与否，x,y代表方框中心，h高度，w宽度，都是相对于整体1的数字，c代表类别。c为（c1，c2......）每个cn都是1或0。而针对一个19×19方格出现多个物体，用anchor boxes解决。假如有80类要探测，且每个方格至多可能出现5个个体。如图所示。
<img src="picture/architecture.png">
</h3>

<h3>2.目标的确定。</h3>
<h3>对于每个anchor box我们找出其最可能的种类。</h3>
<img src="picture/probability_extraction.png">
<h3>如果我们不画出pc为0（无探测物体的的anchor box）。可能得到这样的图片</h3>
<img src="picture/anchor_map.png">
<h3>出现统一物体被多次画框。这时我们要用 IoU和Non-max suppression。就是所谓的交并比和非极大值抑制。iou如图</h3>
<img src="picture/iou.png">
<h3>一般阈值设置为0.5，小于0.5的两个方框为不是一个物体；大于0.5的将被视为两个方框为一个物体，这时就利用非极大值抑制，比较score，得分高者被保留而得分低者去除。</h3>

<h3>3.模型的训练</h3>
<h3>训练需要大量标识探测物体的图片集。所以使用已经训练好的yolo.h5。</h3>

<h2>3.立即使用</h2>
<h3>可以直接对于图片预测车辆yolo_picture.py，参数为输入图片地址和输出图片地址</h3>
<h3>此外对于视频的车辆预测yolo_video.py,参数如下</h3>
<h4>VIDEO_PATH           <br>源视频地址</h4>
<h4>EXTRACT_FOLDER       <br>生成视频存放的位置</h4>
<h4>EXTRACT_FREQUENCY    <br>帧提取频率，每几帧提取一张分析</h4>
<h4>FPS                  <br>形成视频的每秒帧数</h4>
<h4>SIZE                 <br>输出视频大小</h4>








