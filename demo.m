clear all

%% ======================================================================
%  STEP 1: Load data
%
load('dataset/corel.mat');
numExamples = size(data, 2);

result =[];
iCnt = 1;
for sourceFlowerIndex = 1:4
    for sourceTrafficeIndex = 1:3
        if(sourceFlowerIndex == 1)
            trainData = [data(:,1:offset(1)) data(:,offset(4+sourceTrafficeIndex-1)+1:offset(4+sourceTrafficeIndex))];
            trainLabels = [ones(1,offset(1)) zeros(1,offset(4+sourceTrafficeIndex)-offset(4+sourceTrafficeIndex-1))] + 1;
            %%sourceÇÐ¸î1/10×÷unbalance
            trainData = [trainData(:,1:10) data(:,offset(4+sourceTrafficeIndex-1)+1:offset(4+sourceTrafficeIndex))];
            trainLabels = [ones(1,10) zeros(1,offset(4+sourceTrafficeIndex)-offset(4+sourceTrafficeIndex-1))] + 1;
        else
            trainData = [data(:,offset(sourceFlowerIndex-1)+1:offset(sourceFlowerIndex)) data(:,offset(4+sourceTrafficeIndex-1)+1:offset(4+sourceTrafficeIndex))];
            trainLabels = [ones(1,offset(sourceFlowerIndex)-offset(sourceFlowerIndex-1)) zeros(1,offset(4+sourceTrafficeIndex)-offset(4+sourceTrafficeIndex-1))] + 1;
            %%sourceÇÐ¸î1/10×÷unbalance
            trainData = [trainData(:,1:10) data(:,offset(4+sourceTrafficeIndex-1)+1:offset(4+sourceTrafficeIndex))];
            trainLabels =  [ones(1,10) zeros(1,offset(4+sourceTrafficeIndex)-offset(4+sourceTrafficeIndex-1))] + 1;
        end;
        
        for targetFlowerIndex = 1:4
            if targetFlowerIndex == sourceFlowerIndex
                continue
            end
            for targetTrafficIndex = 1:3
                if targetTrafficIndex == sourceTrafficeIndex
                    continue
                end
                
                %% ======================================================================
                %  STEP 1: Load data
                if(targetFlowerIndex == 1)
                    testData = [data(:,1:offset(1)) data(:,offset(4+targetTrafficIndex-1)+1:offset(4+targetTrafficIndex))];
                    testLabels = [ones(1,offset(1)) zeros(1,offset(4+targetTrafficIndex)-offset(4+targetTrafficIndex-1))] + 1;
                    %%targetÇÐ¸î1/10×÷unbalance
                    testData = [testData(:,1:10) data(:,offset(4+sourceTrafficeIndex-1)+1:offset(4+sourceTrafficeIndex))];
                    testLabels = [ones(1,10) zeros(1,offset(4+targetTrafficIndex)-offset(4+targetTrafficIndex-1))] + 1;
                else
                    testData = [data(:,offset(targetFlowerIndex-1)+1:offset(targetFlowerIndex)) data(:,offset(4+targetTrafficIndex-1)+1:offset(4+targetTrafficIndex))];
                    testLabels = [ones(1,offset(targetFlowerIndex)-offset(targetFlowerIndex-1)) zeros(1,offset(4+targetTrafficIndex)-offset(4+targetTrafficIndex-1))] + 1;
                     %%targetÇÐ¸î1/10×÷unbalance
                    testData = [testData(:,1:10) data(:,offset(4+targetTrafficIndex-1)+1:offset(4+targetTrafficIndex))];
                    testLabels = [ones(1,10) zeros(1,offset(4+targetTrafficIndex)-offset(4+targetTrafficIndex-1))] + 1;
                end;
                
                %% ======================================================================
                %  STEP 2: Sparse Autoencoder
                numK = 10;  % number of hidden units 
                numC = 2;
                numM = size(trainData,1);                % input data feature dimensions

                theta = initialize_img1(numK, numM, numC, trainData, testData);    % Randomly initialize the parameters     
                addpath minFunc/
                
                options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost function. 
                options.maxIter = 300;	  % Maximum number of iterations of L-BFGS to run 
                options.display = 'on';
                options.TolFun  = 1e-6;
                options.TolX = 1e-1119;
                options.maxFunEvals = 4000;
                label1 = [trainData;ones(1,size(trainLabels,2))-trainLabels];
                alpha = 0.005;%1; 
                lambda = 0.1;

                [opttheta, cost] = minFunc( @(p) computeSparse(p, numM, numK,...
                            numC, alpha, trainData, testData), theta, options);  

                 % get W1 W11 b1 b11 after training
                W1 = reshape(opttheta(1:numK*numM), numK, numM);
                b1 = opttheta(2*numK*numM+2*numK*numC+1:2*numK*numM+2*numK*numC+numK);
                W11 = reshape(opttheta(numK*numM+2*numK*numC+1:2*numK*numM+2*numK*numC), numM, numK);
                b11 = opttheta(2*numK*numM+2*numK*numC+2*numK+numC+1:2*numK*numM+2*numK*numC+2*numK+numC+numM);
                
                 %% ======================================================================
                %  STEP 3
               TwoRoundTrainData = sigmoid(W1 * trainData + b1 * ones(1, size(trainData,2)));
               TwoRoundTrainData = sigmoid(W11 * TwoRoundTrainData + b11 * ones(1, size(TwoRoundTrainData,2)));
               TwoRoundTestData = sigmoid(W1 * testData + b1 * ones(1, size(testData,2)));
               TwoRoundTestData = sigmoid(W11 * TwoRoundTestData + b11 * ones(1, size(TwoRoundTestData,2)));
               
                numK = 10;  % number of hidden units 
                numC = 2;
                numM = size(trainData,1);                % input data feature dimensions
                pos = find(trainLabels==2);
                neg = find(trainLabels==1);
                numX = [size(pos,2), size(neg,2)];
  
               beta = 2;%0.5; 
               gamma = 0.0001;
                options.maxIter = 50;	  % Maximum number of iterations of L-BFGS to run 
                options.display = 'on';
                options.TolFun  = 1e-3;
                options.TolX = 1e-1119;
                options.maxFunEvals = 4000;
                options.M = MMD(TwoRoundTrainData, TwoRoundTestData, trainLabels, testLabels);
               [opttheta, cost] = minFunc( @(p) computeObjectAndGradiend(p, numM, numK,...
                            numC, numX, beta, gamma, TwoRoundTrainData, TwoRoundTestData), theta, options);  
               
               W1 = reshape(theta(1:numK*numM), numK, numM);
               W2 = reshape(theta(numK*numM+1:numK*numM+numK*numC), numC, numK);
               b1 = theta(2*numK*numM+2*numK*numC+1:2*numK*numM+2*numK*numC+numK);
               b2 = theta(2*numK*numM+2*numK*numC+numK+1:2*numK*numM+2*numK*numC+numK+numC);
                
                hiddeninputs_train = sigmoid(W1 * TwoRoundTrainData + b1 * ones(1, size(TwoRoundTrainData,2)));
                hiddeninputs_test = sigmoid(W1 * TwoRoundTestData + b1 * ones(1, size(TwoRoundTestData,2)));
                after_accuracy_LR = test_LR(hiddeninputs_train, hiddeninputs_test, trainLabels, testLabels);
                after_accuracy = test(trainData, testData, trainLabels, testLabels, W1, b1, W2, b2);
                iCnt = iCnt + 1;
            end
        end
    end
end