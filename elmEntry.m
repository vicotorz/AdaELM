function [finalresult] = elmEntry(TrainingData_File,TrainTopicNumber,TestingData_File,TestTopicNumber,Elm_Type, NumberofHiddenNeurons, ActivationFunction)
%输入的命令：[finalresult] = elmEntry('Train.txt',10,'Test.txt ',10,1,500,'sig')

%%%%%%%%%%% Load training dataset
%载入训练集
train_data=load(TrainingData_File);
T=train_data(:,1:TrainTopicNumber);%对应矩阵的第一列 ‘代表转置
P=train_data(:,TrainTopicNumber+1:size(train_data,2));%除了第一列的全部
clear train_data;   %拆分并删除原始矩阵

%%%%%%%%%%% Load testing dataset
%载入测试集
test_data=load(TestingData_File);
if TestTopicNumber>0
    TV.T=test_data(:,1:TestTopicNumber);%分列
    TV.P=test_data(:,TestTopicNumber+1:size(test_data,2));%分列
else
    TV.T=ones(size(test_data,1),1);%创建列
    TV.P=test_data(:,1:size(test_data,2));%分列
    fprintf('chuangjian');
end
clear test_data;
%size(TV.T);
%size(TV.P);
%拼接矩阵
y=1;
finalresult=zeros(TestTopicNumber,size(TV.T,1));
for i=1:TrainTopicNumber
    trainmatrix=[T(:,i),P];
    
    if TestTopicNumber>0
        testmatrix=[TV.T(:,i),TV.P];
    else
        testmatrix=[TV.T,TV.P];
    end
   
    TEST_SIZE=size(testmatrix)
    [TrainingTime, TrainingAccuracy,fin_TY] = elm_adaboost(trainmatrix,testmatrix,TestTopicNumber,Elm_Type, NumberofHiddenNeurons, ActivationFunction);
                                                      %elm_adaboost(TrainingData_File,TestingData_File,Elm_Type, NumberofHiddenNeurons, ActivationFunction)
    fprintf('训练--预测完成!');
    for i = 1 : size(TV.T, 1)
        %SIZE=size(test_sum_output(:,i))
        finalresult(y,i)=fin_TY(i,1);
    end   
    y=y+1;
    finalresult

end
 
fid = fopen('result.txt', 'wt');
[m, n] = size(finalresult);
 for i = 1 : m
    for j = 1 : n % 逐行打印出来
        fprintf(fid, '%d ', finalresult(i, j)); % 注意%f后面有一个空格
     end
     fprintf(fid, '\n');
 end
end