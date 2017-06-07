function [TrainingTime, TrainingAccuracy,fin_TY] = elm_adaboost(TrainingData_File,TestingData_File,TestTopicNumber,Elm_Type, NumberofHiddenNeurons, ActivationFunction)
% [TrainingTime,TrainingAccuracy,TestingAccuracy] = elm_adaboost('test.sgm','N.sgm ',1,500,'sig')
%ELM[训练集，测试集，elm类型，隐层神经元数量，激活函数]
% [TrainingTime,TrainingAccuracy,TestingAccuracy] = elm_adaboost('train_fin.sgm','test_fin.sgm ',1,500,'sig')

% Usage: elm(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction)
% OR:    [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction)
%
% Input:
% TrainingData_File     - Filename of training data set
% TestingData_File      - Filename of testing data set
% Elm_Type              - 0 for regression; 1 for (both binary and multi-classes) classification
% NumberofHiddenNeurons - Number of hidden neurons assigned to the ELM
% ActivationFunction    - Type of activation function:
%                           'sig' for Sigmoidal function
%                           'sin' for Sine function
%                           'hardlim' for Hardlim function
%                           'tribas' for Triangular basis function
%                           'radbas' for Radial basis function (for additive type of SLFNs instead of RBF type of SLFNs)
%
% Output: 
% TrainingTime          - Time (seconds) spent on training ELM
% TestingTime           - Time (seconds) spent on predicting ALL testing data
% TrainingAccuracy      - Training accuracy: 
%                           RMSE for regression or correct classification rate for classification
% TestingAccuracy       - Testing accuracy: 
%                           RMSE for regression or correct classification rate for classification
%
% MULTI-CLASSE CLASSIFICATION: NUMBER OF OUTPUT NEURONS WILL BE AUTOMATICALLY SET EQUAL TO NUMBER OF CLASSES
% FOR EXAMPLE, if there are 7 classes in all, there will have 7 output
% neurons; neuron 5 has the highest output means input belongs to 5-th class
%
% Sample1 regression: [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = elm('sinc_train', 'sinc_test', 0, 20, 'sig')
% Sample2 classification: elm('diabetes_train', 'diabetes_test', 1, 20, 'sig')
%
    %%%%    Authors:    MR QIN-YU ZHU AND DR GUANG-BIN HUANG
    %%%%    NANYANG TECHNOLOGICAL UNIVERSITY, SINGAPORE
    %%%%    EMAIL:      EGBHUANG@NTU.EDU.SG; GBHUANG@IEEE.ORG
    %%%%    WEBSITE:    http://www.ntu.edu.sg/eee/icis/cv/egbhuang.htm
    %%%%    DATE:       APRIL 2004

%%%%%%%%%%% Macro definition
%宏定义
REGRESSION=0;
CLASSIFIER=1;
%LABEL_KIND=136;%!Routers-21578有效标签有135个，再加上没有标签一共136个

%%%%%%%%%%% Load training dataset
%载入训练集
T=TrainingData_File(:,1)';%对应矩阵的第一列 ‘代表转置
P=TrainingData_File(:,2:size(TrainingData_File,2))';%除了第一列的全部
                                  %   Release raw training data array  拆分并删除原始矩阵

%%%%%%%%%%% Load testing dataset
%载入测试集
if TestTopicNumber>0
    fprintf('1');
    TV.T=TestingData_File(:,1)';
    TV.P=TestingData_File(:,2:size(TestingData_File,2))';%除了第一列的全部
else
    fprintf('2');
    TV.T=ones(size(TestingData_File,1),1)';
    TV.P=TestingData_File(:,2:size(TestingData_File,2))';%除了第一列的全部
    %RRRR=size(TestingData_File)
    %WHAT=size(TV.P)
    
end
clear test_data;                                    %   Release raw testing data array   拆分并删除原始矩阵

NumberofTrainingData=size(P,2);%要训练的指标总数
NumberofTestingData=size(TV.P,2);%要检测的指标总数
NumberofInputNeurons=size(P,1);%要输入的总数（节点数）

W_orgin=1/size(T,2);%初始权值
for i=1:size(T,2)
   DM(1,i)=1/size(T,2);%初始权值%对应的权值矩阵
end

if Elm_Type~=REGRESSION%REGRESSION --分类
    %%%%%%%%%%%% Preprocessing the data of classification
    %确定输出节点的个数
    sorted_target=sort(cat(2,T,TV.T),2);%cat是按照2维连接矩阵T和TV.T。并且排序
    label=zeros(1,1);                               %   Find and save in 'label' class label from training and testing data sets
    label(1,1)=sorted_target(1,1);
    j=1;
    for i = 2:(NumberofTrainingData+NumberofTestingData)
        if sorted_target(1,i) ~= label(1,j)
            j=j+1;
            label(1,j) = sorted_target(1,i);
        end
    end
    number_class=j;
    NumberofOutputNeurons=number_class;%确定最终输出节点的个数
       
    %%%%%%%%%% Processing the targets of training训练分配标签
    temp_T=zeros(NumberofOutputNeurons, NumberofTrainingData);
    for i = 1:NumberofTrainingData
        for j = 1:number_class
            if label(1,j) == T(1,i)
                break; 
            end
        end
        temp_T(j,i)=1;
    end
    T=temp_T*2-1;

    %%%%%%%%%% Processing the targets of testing测试分配标签
    temp_TV_T=zeros(NumberofOutputNeurons, NumberofTestingData);
    for i = 1:NumberofTestingData
        for j = 1:number_class
            if label(1,j) == TV.T(1,i)
                break; 
            end
        end
        temp_TV_T(j,i)=1;
    end
    TV.T=temp_TV_T*2-1;
end                                                 %   end if of Elm_Type

%%%%%%%%%%% Calculate weights & biases 计算权值和偏置
start_time_train=cputime;
index=1;%用于输入输出的建立------------不能低于10个
while index<=10
%%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
%随机生成权重和偏置和隐层神经元
InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;%输入权重（随机分配的）
BiasofHiddenNeurons=rand(NumberofHiddenNeurons,1);%隐层神经元
tempH=InputWeight*P;%1）按照公式乘积
%clear P;                                            %   Release input of training data 
ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH=tempH+BiasMatrix;%2）加偏置阈值

%%%%%%%%%%% Calculate hidden neuron output matrix H 计算隐层输出矩阵
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H = 1 ./ (1 + exp(-tempH));
    case {'sin','sine'}
        %%%%%%%% Sine
        H = sin(tempH);    
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H = double(hardlim(tempH));
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H = tribas(tempH);
    case {'radbas'}
        %%%%%%%% Radial basis function
        H = radbas(tempH);
        %%%%%%%% More activation functions can be added here                
end
clear tempH;                                        %   Release the temparary array for calculation of hidden neuron output matrix H
%所以到这里H是隐层的输出矩阵

%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
%计算输出权重
OutputWeight=pinv(H') * T';    %pinv指的是矩阵的逆矩阵  % implementation without regularization factor //refer to 2006 Neurocomputing paper
%OutputWeight=inv(eye(size(H,1))/C+H * H') * H * T';   % faster method 1 //refer to 2012 IEEE TSMC-B paper
%implementation; one can set regularizaiton factor C properly in classification applications 
%OutputWeight=(eye(size(H,1))/C+H * H') \ H * T';      % faster method 2 //refer to 2012 IEEE TSMC-B paper
%implementation; one can set regularizaiton factor C properly in classification applications
%If you use faster methods or kernel method, PLEASE CITE in your paper properly: 
%Guang-Bin Huang, Hongming Zhou, Xiaojian Ding, and Rui Zhang, "Extreme Learning Machine for Regression and Multi-Class Classification," submitted to IEEE Transactions on Pattern Analysis and Machine Intelligence, October 2010. 

%%%%%%%%%%% Calculate the training accuracy
Y=(H' * OutputWeight)';                             %   Y: the actual output of the training data
%Y是实际输出，隐层输出*输出权重=输出矩阵
if Elm_Type == REGRESSION
    TrainingAccuracy=sqrt(mse(T - Y))%mse平方               %   Calculate training accuracy (RMSE) for regression case
end
clear H;
%111%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%111%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
em=0;
sum=0;
for i = 1 : size(T,2)
        [x, label_index_expected]=max(T(:,i));%T表示标签和文本的矩阵
        [x, label_index_actual]=max(Y(:,i));%找到列里面最大的那个
        if label_index_actual~=label_index_expected
            %错误点计数
            sum=sum+1;
            em=em+DM(1,i);
        end
end

if em>=0.5%必须保证正数
    continue%!用continue控制下面的过程
else

ALL_INPUT{1,index}=InputWeight;%记录输入权重
ALL_OUTPUT{1,index}=OutputWeight;%记录输出矩阵
am=log((1-em)/em);%对应弱分类器权值%如果希望权值是大于0的，那么必须满足0~0.5之间
ALL_AM{1,index}=am;
%更新标签
%权值全部都需要更新
for i=1:size(T,2)
    [x, label_index_expected]=max(T(:,i));%T表示标签和文本的矩阵
    [x, label_index_actual]=max(Y(:,i));%找到列里面最大的那个
    if label_index_actual~=label_index_expected
      DM(1,i)=DM(1,i)*am;%更改出错点权值(这里可能有问题)
    end
end
Zm=0;
%归一化
for i=1:size(T,2)
    Zm=Zm+DM(1,i);
end
for i=1:size(T,2)
    DM(1,i)=DM(1,i)/Zm;
end
%222%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%222%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
index=index+1;
end
end

end_time_train=cputime;
TrainingTime=end_time_train-start_time_train        %   Calculate CPU time (seconds) spent for training ELM
fprintf('训练结束');
ALL_AM

if Elm_Type == REGRESSION
    TestingAccuracy=sqrt(mse(TV.T - TY))            %   Calculate testing accuracy (RMSE) for regression case
end

if Elm_Type == CLASSIFIER
%%%%%%%%%% Calculate training & testing classification accuracy

    MissClassificationRate_Training=0;
    MissClassificationRate_Testing=0;

    for i = 1 : size(T, 2)
        [x, label_index_expected]=max(T(:,i));
        [x, label_index_actual]=max(Y(:,i));
        if label_index_actual~=label_index_expected
            MissClassificationRate_Training=MissClassificationRate_Training+1;
        end
    end
    TrainingAccuracy=1-MissClassificationRate_Training/size(T,2)
   
%111%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%111%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
number=size(ALL_AM,2);
start_time_test=cputime;
size(Y)
vote_matrix=zeros(size(TV.T,2),size(Y,1));%投票标签矩阵

for in=1:number
    %SIZE_INPUT=size(ALL_INPUT{1,in})
    %SIZE_TVP=size(TV.P)
tempH_test=ALL_INPUT{1,in}*TV.P;%输入矩阵乘输入权重
%clear TV.P;             %   Release input of testing data             
ind=ones(1,NumberofTestingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH_test=tempH_test + BiasMatrix;%再加上偏置
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H_test = 1 ./ (1 + exp(-tempH_test));
    case {'sin','sine'}
        %%%%%%%% Sine
        H_test = sin(tempH_test);        
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H_test = hardlim(tempH_test);        
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H_test = tribas(tempH_test);        
    case {'radbas'}
        %%%%%%%% Radial basis function
        H_test = radbas(tempH_test);        
        %%%%%%%% More activation functions can be added here        
end
TY=(H_test'* ALL_OUTPUT{1,in})';
%弱分类器这样乘权值了，应该再有个矩阵，记录该文本所属标签的最大可能性，并进行投票    文本个数size(TV.T,2)*标签总数（LABEL_KIND）
%!TY的标签是经过归一化处理的，不会出现-1的情况
%input('')
for g=1:size(TV.T,2)
    [x, label_index_expected]=max(TY(:,g)); 
    %if g==2
        %label_index_expected
    %end
     vote_matrix(g,label_index_expected)= vote_matrix(g,label_index_expected)+ALL_AM{1,in};    
end
%test_sum_output=test_sum_output+ TY*ALL_AM{1,in};%算上了激励函数
end
end_time_test=cputime;
TestingTime=end_time_test-start_time_test;  
%最后需要整合统计vote_matrix标签形成一个多行，一列的矩阵
%INF表示“无穷大”
%NAN表示“无效数字”
fin_TY=zeros(size(TY,2),1);
for i=1:size(TY,2)
    %获得某一行最大值对应的列[my_max,col]=max(A(2,:))
    [my_max,col]=max(vote_matrix(i,:));
    %TY(i,1)=col;  
    fin_TY(i,1)=col;    
end
%222%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%222%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%加入查全率（召回率）
%统计各个标签的总数
%统计有多少个标签
la=zeros(1,size(TV.T,2));
for o=1:size(TV.T,2)
 [x, label_index_expected]=max(TV.T(:,o));
 la(1,o)=label_index_expected;
end

B=[];
name=union(la,B);
%name
total_label_num=size(name,2);
lab_total=zeros(1,total_label_num);
%start_loc=1;
%f=0;
for t=1:size(TV.T,2)
    [x, label_index_expected]=max(TV.T(:,t));
   for y=1:total_label_num
   if name(1,y)==label_index_expected
       lab_total(1,y)=lab_total(1,y)+1;
      % f=1;
   end   
   end
end   

%lab_total  和 name分别记录着每个标签和总数
result=zeros(1,total_label_num);
    for i = 1 : size(TV.T, 2)
        %[x, label_index_expected]=max(TV.T(:,i));
        label_index_expected=fin_TY(i,1);
        [x, label_index_actual]=max(TY(:,i));
        
        if label_index_actual~=label_index_expected
            for u=1:total_label_num
                if name(1,u)==label_index_actual
                    result(1,u)=result(1,u)+1;
                end
            end           
            MissClassificationRate_Testing=MissClassificationRate_Testing+1;
        end
        
    end
    TestingAccuracy=1-MissClassificationRate_Testing/size(TV.T,2);
    recall=0;
    recall2=0
    
    for f=1:total_label_num
        1-result(1,f)/lab_total(1,f)
        recall=recall+(lab_total(1,f)/size(TV.T,2))*(1-(result(1,f)/lab_total(1,f)));
        recall2=recall2+(1-(result(1,f)/lab_total(1,f)));
    end
    %recall
    %recall2
end