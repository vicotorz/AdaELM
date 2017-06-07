function [TrainingTime, TrainingAccuracy,fin_TY] = elm_adaboost(TrainingData_File,TestingData_File,TestTopicNumber,Elm_Type, NumberofHiddenNeurons, ActivationFunction)
% [TrainingTime,TrainingAccuracy,TestingAccuracy] = elm_adaboost('test.sgm','N.sgm ',1,500,'sig')
%ELM[ѵ���������Լ���elm���ͣ�������Ԫ�����������]
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
%�궨��
REGRESSION=0;
CLASSIFIER=1;
%LABEL_KIND=136;%!Routers-21578��Ч��ǩ��135�����ټ���û�б�ǩһ��136��

%%%%%%%%%%% Load training dataset
%����ѵ����
T=TrainingData_File(:,1)';%��Ӧ����ĵ�һ�� ������ת��
P=TrainingData_File(:,2:size(TrainingData_File,2))';%���˵�һ�е�ȫ��
                                  %   Release raw training data array  ��ֲ�ɾ��ԭʼ����

%%%%%%%%%%% Load testing dataset
%������Լ�
if TestTopicNumber>0
    fprintf('1');
    TV.T=TestingData_File(:,1)';
    TV.P=TestingData_File(:,2:size(TestingData_File,2))';%���˵�һ�е�ȫ��
else
    fprintf('2');
    TV.T=ones(size(TestingData_File,1),1)';
    TV.P=TestingData_File(:,2:size(TestingData_File,2))';%���˵�һ�е�ȫ��
    %RRRR=size(TestingData_File)
    %WHAT=size(TV.P)
    
end
clear test_data;                                    %   Release raw testing data array   ��ֲ�ɾ��ԭʼ����

NumberofTrainingData=size(P,2);%Ҫѵ����ָ������
NumberofTestingData=size(TV.P,2);%Ҫ����ָ������
NumberofInputNeurons=size(P,1);%Ҫ������������ڵ�����

W_orgin=1/size(T,2);%��ʼȨֵ
for i=1:size(T,2)
   DM(1,i)=1/size(T,2);%��ʼȨֵ%��Ӧ��Ȩֵ����
end

if Elm_Type~=REGRESSION%REGRESSION --����
    %%%%%%%%%%%% Preprocessing the data of classification
    %ȷ������ڵ�ĸ���
    sorted_target=sort(cat(2,T,TV.T),2);%cat�ǰ���2ά���Ӿ���T��TV.T����������
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
    NumberofOutputNeurons=number_class;%ȷ����������ڵ�ĸ���
       
    %%%%%%%%%% Processing the targets of trainingѵ�������ǩ
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

    %%%%%%%%%% Processing the targets of testing���Է����ǩ
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

%%%%%%%%%%% Calculate weights & biases ����Ȩֵ��ƫ��
start_time_train=cputime;
index=1;%������������Ľ���------------���ܵ���10��
while index<=10
%%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
%�������Ȩ�غ�ƫ�ú�������Ԫ
InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;%����Ȩ�أ��������ģ�
BiasofHiddenNeurons=rand(NumberofHiddenNeurons,1);%������Ԫ
tempH=InputWeight*P;%1�����չ�ʽ�˻�
%clear P;                                            %   Release input of training data 
ind=ones(1,NumberofTrainingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH=tempH+BiasMatrix;%2����ƫ����ֵ

%%%%%%%%%%% Calculate hidden neuron output matrix H ���������������
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
%���Ե�����H��������������

%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
%�������Ȩ��
OutputWeight=pinv(H') * T';    %pinvָ���Ǿ���������  % implementation without regularization factor //refer to 2006 Neurocomputing paper
%OutputWeight=inv(eye(size(H,1))/C+H * H') * H * T';   % faster method 1 //refer to 2012 IEEE TSMC-B paper
%implementation; one can set regularizaiton factor C properly in classification applications 
%OutputWeight=(eye(size(H,1))/C+H * H') \ H * T';      % faster method 2 //refer to 2012 IEEE TSMC-B paper
%implementation; one can set regularizaiton factor C properly in classification applications
%If you use faster methods or kernel method, PLEASE CITE in your paper properly: 
%Guang-Bin Huang, Hongming Zhou, Xiaojian Ding, and Rui Zhang, "Extreme Learning Machine for Regression and Multi-Class Classification," submitted to IEEE Transactions on Pattern Analysis and Machine Intelligence, October 2010. 

%%%%%%%%%%% Calculate the training accuracy
Y=(H' * OutputWeight)';                             %   Y: the actual output of the training data
%Y��ʵ��������������*���Ȩ��=�������
if Elm_Type == REGRESSION
    TrainingAccuracy=sqrt(mse(T - Y))%mseƽ��               %   Calculate training accuracy (RMSE) for regression case
end
clear H;
%111%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%111%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
em=0;
sum=0;
for i = 1 : size(T,2)
        [x, label_index_expected]=max(T(:,i));%T��ʾ��ǩ���ı��ľ���
        [x, label_index_actual]=max(Y(:,i));%�ҵ������������Ǹ�
        if label_index_actual~=label_index_expected
            %��������
            sum=sum+1;
            em=em+DM(1,i);
        end
end

if em>=0.5%���뱣֤����
    continue%!��continue��������Ĺ���
else

ALL_INPUT{1,index}=InputWeight;%��¼����Ȩ��
ALL_OUTPUT{1,index}=OutputWeight;%��¼�������
am=log((1-em)/em);%��Ӧ��������Ȩֵ%���ϣ��Ȩֵ�Ǵ���0�ģ���ô��������0~0.5֮��
ALL_AM{1,index}=am;
%���±�ǩ
%Ȩֵȫ������Ҫ����
for i=1:size(T,2)
    [x, label_index_expected]=max(T(:,i));%T��ʾ��ǩ���ı��ľ���
    [x, label_index_actual]=max(Y(:,i));%�ҵ������������Ǹ�
    if label_index_actual~=label_index_expected
      DM(1,i)=DM(1,i)*am;%���ĳ����Ȩֵ(�������������)
    end
end
Zm=0;
%��һ��
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
fprintf('ѵ������');
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
vote_matrix=zeros(size(TV.T,2),size(Y,1));%ͶƱ��ǩ����

for in=1:number
    %SIZE_INPUT=size(ALL_INPUT{1,in})
    %SIZE_TVP=size(TV.P)
tempH_test=ALL_INPUT{1,in}*TV.P;%������������Ȩ��
%clear TV.P;             %   Release input of testing data             
ind=ones(1,NumberofTestingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH_test=tempH_test + BiasMatrix;%�ټ���ƫ��
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
%��������������Ȩֵ�ˣ�Ӧ�����и����󣬼�¼���ı�������ǩ���������ԣ�������ͶƱ    �ı�����size(TV.T,2)*��ǩ������LABEL_KIND��
%!TY�ı�ǩ�Ǿ�����һ������ģ��������-1�����
%input('')
for g=1:size(TV.T,2)
    [x, label_index_expected]=max(TY(:,g)); 
    %if g==2
        %label_index_expected
    %end
     vote_matrix(g,label_index_expected)= vote_matrix(g,label_index_expected)+ALL_AM{1,in};    
end
%test_sum_output=test_sum_output+ TY*ALL_AM{1,in};%�����˼�������
end
end_time_test=cputime;
TestingTime=end_time_test-start_time_test;  
%�����Ҫ����ͳ��vote_matrix��ǩ�γ�һ�����У�һ�еľ���
%INF��ʾ�������
%NAN��ʾ����Ч���֡�
fin_TY=zeros(size(TY,2),1);
for i=1:size(TY,2)
    %���ĳһ�����ֵ��Ӧ����[my_max,col]=max(A(2,:))
    [my_max,col]=max(vote_matrix(i,:));
    %TY(i,1)=col;  
    fin_TY(i,1)=col;    
end
%222%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%222%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%�����ȫ�ʣ��ٻ��ʣ�
%ͳ�Ƹ�����ǩ������
%ͳ���ж��ٸ���ǩ
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

%lab_total  �� name�ֱ��¼��ÿ����ǩ������
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