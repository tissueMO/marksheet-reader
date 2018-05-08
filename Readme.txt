◆操作方法
1. ビッグデータ分析用サーバー (192.168.211.61) に SSH @tfuser で入る
2. 以下、操作を行う

###############################################################

[tfuser@tensor1 omr]$ cd /home/tfuser/tensor/omr/

[tfuser@tensor1 omr]$ source activate tensorflow

(tensorflow) [tfuser@tensor1 omr]$ python marksheet_reader.py --imgdir image/ --verbose
image/scan_src.jpg
Q-01. [3]
Q-02. [2]
Q-03. [5]
Q-04. [4]
Q-05. [1 3]  # 複数回答 #
Q-06. ** 未回答 **
image/scan_src2.jpg
Q-01. [3]
Q-02. [2]
Q-03. [5]
Q-04. [4]
Q-05. [1 3]  # 複数回答 #
Q-06. ** 未回答 **
image/scan_src3.jpg
Q-01. [3]
Q-02. [2]
Q-03. [5]
Q-04. [4]
Q-05. [1 3]  # 複数回答 #
Q-06. ** 未回答 **
image/marker.jpg
マーカーの認識に失敗

◆集計結果
   Q-No.  Ans-1  Ans-2  Ans-3  Ans-4  Ans-5
0   Q-1      0      3      0      0      0
1   Q-2      3      0      0      0      0
2   Q-3      0      0      0      3      0
3   Q-4      0      0      3      0      0
4   Q-5      0      0      0      0      0
5   Q-6      0      0      0      0      0

◆複数回答
            ファイル名  設問番号    答え？
0   scan_src.jpg   5.0  [1 3]
1  scan_src2.jpg   5.0  [1 3]
2  scan_src3.jpg   5.0  [1 3]

◆無回答
            ファイル名  設問番号
0   scan_src.jpg   6.0
1  scan_src2.jpg   6.0
2  scan_src3.jpg   6.0

◆認識エラー
         ファイル名
0  marker.jpg


集計結果を { summary 以下 } に書き出しました

###############################################################
