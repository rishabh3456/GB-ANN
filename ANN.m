%% IMPORT DATA

R = readmatrix('TestDataSheet.csv');
x=R(:,[1:3,10,14,15])'; % input
t=R(:,7:9)'; % target
[ I N ] = size(x); % [ 6 178 ]
[ O N ]  = size(t); % [ 3 178 ]




%% CREATE NETWORK AND SET PARAMETERS

fractionTrain = 70/100; % How much of data will go to training
fractionValid = 15/100; % How much of data will go to validation
fractionTest = 15/100; % How much of data will go to testing
Ntrn = N - round(fractionTest*N + fractionValid*N); % Number of training examples
Ntrneq = Ntrn*O; % Number of training equations
% MSE for a naïve constant output model that always outputs average of target data
MSE00a = mean(var(t',0));
%%Find a good value for number of neurons H in hidden layer
Hmin = 1; % Lowest number of neurons H to consider (a linear model)
% To avoid overfitting with too many neurons, require that Ntrneq > Nw==> H <= Hub (upper bound)
Hub = -1+ceil( (Ntrneq-O) / ( I+O+1) ); % Upper bound for Ntrneq >= Nw
Hmax = Hub;
dH = 1;
 rng(1);
% Number of random initial weight trials for each candidate h
Ntrials = 10;
perf=0; % Any large value
hbest=10000000;
for h=Hmin:dH:Hmax % Number of neurons in hidden layer
    net = fitnet(h); % Create a MLP
    Nw = (I+1)*h + (h+1)*O; % Number of unknown weights
    Ntrndof = Ntrneq - Nw; % Degrees of Freedom
    net.trainParam.showWindow= false; % Don't oppen nntraintool
    net.divideParam.trainRatio = fractionTrain; % Training set size
    net.divideParam.testRatio = fractionTest; % Test set size
    net.divideParam.valRatio = fractionValid; % Validation set size
    net.trainParam.epochs = 100; % Number of epochs
    net.layers{1}.transferFcn = 'tansig'; % Set the activation function
    net.performFcn = 'mse';  % Mean Squared Error for testing performance
    
%% TRAIN NETWORK
    for i=1:Ntrials
          % Configure network inputs and outputs to best match input and target data
          net = configure(net, x, t);
          [net, tr, ygs, e] = train(net,x,t); % Train the net
          y = net(x); % Get output values from net
          perftemp = 1-mse(e)/MSE00a; % Compare output with targets using percentage mse
          if(perftemp>=0.98 && h<=hbest )
              if(perftemp>perf) % Find best performing net
                  hbest=h;
                  perf=perftemp; % Choose least mse
                  finalnet=net; % Choose best net as finalnet
                  finaly=y; % Choose best net output
                  trbest=tr; % Choose best net training record
              end
          end
    end
end
%  fprintf('Actual Error: %f percent\n',perf*100); % Print best percentage mse

%% MSE AND R^2 FOR EACH OUTPUT

%Initializations
testInput=[];
testData=[];
trainInput=[];
trainData=[];
valInput=[];
valData=[];

for i=1:size(tr.testMask{1},2) % Iterate over all values of the dataset
    if (~isnan(tr.trainMask{1}(1,i))) % For all values present in train set 
        trainInput(:,end+1)=x(:,i); % Get the train set inputs
        trainData(:,end+1)=t(:,i); % Get the train set targets
    elseif (~isnan(tr.testMask{1}(1,i))) % For all values present in test set
        testInput(:,end+1)=x(:,i); % Get the test set inputs
        testData(:,end+1)=t(:,i); % Get the test set targets
    else % For all values present in val set
        valInput(:,end+1)=x(:,i); % Get the val set inputs
        valData(:,end+1)=t(:,i); % Get the val set targets
    end
end
trains=finalnet(trainInput);
test=finalnet(testInput);
val=finalnet(valInput);
msew=sqrt(mean((test(1,:)-testData(1,:)).^2));
msed=sqrt(mean((test(2,:)-testData(2,:)).^2));
mseh=sqrt(mean((test(3,:)-testData(3,:)).^2));
r2w=1-sum((test(1,:)-testData(1,:)).^2)/sum((mean(testData(1,:))-testData(1,:)).^2);
r2d=1-sum((test(2,:)-testData(2,:)).^2)/sum((mean(testData(2,:))-testData(2,:)).^2);
r2h=1-sum((test(3,:)-testData(3,:)).^2)/sum((mean(testData(3,:))-testData(3,:)).^2);
fprintf("Width Error: %f \nDepth error: %f \nHeight error: %f\n",msew,msed,mseh);
fprintf("Width R^2: %f \nDepth R^2: %f \nHeight R^2: %f\n",r2w,r2d,r2h);

%% PLOTS

figure
plotperform(trbest) % Performance Plot

figure
plotregression(trainData,trains,"Train",testData,test,"Test",valData,val,"Validation",t,finaly,"Overall") % Regression Plots
