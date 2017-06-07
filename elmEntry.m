function [finalresult] = elmEntry(TrainingData_File,TrainTopicNumber,TestingData_File,TestTopicNumber,Elm_Type, NumberofHiddenNeurons, ActivationFunction)
%��������[finalresult] = elmEntry('Train.txt',10,'Test.txt ',10,1,500,'sig')

%%%%%%%%%%% Load training dataset
%����ѵ����
train_data=load(TrainingData_File);
T=train_data(:,1:TrainTopicNumber);%��Ӧ����ĵ�һ�� ������ת��
P=train_data(:,TrainTopicNumber+1:size(train_data,2));%���˵�һ�е�ȫ��
clear train_data;   %��ֲ�ɾ��ԭʼ����

%%%%%%%%%%% Load testing dataset
%������Լ�
test_data=load(TestingData_File);
if TestTopicNumber>0
    TV.T=test_data(:,1:TestTopicNumber);%����
    TV.P=test_data(:,TestTopicNumber+1:size(test_data,2));%����
else
    TV.T=ones(size(test_data,1),1);%������
    TV.P=test_data(:,1:size(test_data,2));%����
    fprintf('chuangjian');
end
clear test_data;
%size(TV.T);
%size(TV.P);
%ƴ�Ӿ���
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
    fprintf('ѵ��--Ԥ�����!');
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
    for j = 1 : n % ���д�ӡ����
        fprintf(fid, '%d ', finalresult(i, j)); % ע��%f������һ���ո�
     end
     fprintf(fid, '\n');
 end
end