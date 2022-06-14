%% ======================================================================
%  STEP 0: parameters

addpath('C:\Users\pc\Documents\MATLAB\unbalance\method\')
addpath('C:\Users\pc\Documents\MATLAB\unbalance\examples\libsvm-3.11\matlab\');

global params;
global maniParameter;


% global numX;               % number of positive and negative instances in traindata

%% ======================================================================
%  STEP 1: Load data
load('C:\Users\pc\Documents\MATLAB\unbalance\dataset\corel.mat');
numExamples = size(data, 2);

Result = [] ;
iCnt = 1;
for sourceFlowerIndex = 1:4
    for sourceTrafficeIndex = 1:4
        if(sourceFlowerIndex == 1)
            trainData = [data(:,1:offset(1)) data(:,offset(4+sourceTrafficeIndex-1)+1:offset(4+sourceTrafficeIndex))];
            trainLabels = [ones(1,offset(1)) zeros(1,offset(4+sourceTrafficeIndex)-offset(4+sourceTrafficeIndex-1))] + 1;
            numX = [offset(1), offset(4+sourceTrafficeIndex)-offset(4+sourceTrafficeIndex-1)];
        else
            trainData = [data(:,offset(sourceFlowerIndex-1)+1:offset(sourceFlowerIndex)) data(:,offset(4+sourceTrafficeIndex-1)+1:offset(4+sourceTrafficeIndex))];
            trainLabels = [ones(1,offset(sourceFlowerIndex)-offset(sourceFlowerIndex-1)) zeros(1,offset(4+sourceTrafficeIndex)-offset(4+sourceTrafficeIndex-1))] + 1;
            numX = [offset(sourceFlowerIndex)-offset(sourceFlowerIndex-1), offset(4+sourceTrafficeIndex)-offset(4+sourceTrafficeIndex-1)];
        end;
            
        for targetFlowerIndex = 1:4
            if targetFlowerIndex == sourceFlowerIndex
                continue
            end
            for targetTrafficIndex = 1:4
                if targetTrafficIndex == sourceTrafficeIndex
                    continue
                end
                
                imgSize = 16;
                params.patchWidth=9;           % width of a patch
                params.n=params.patchWidth^2;   % dimensionality of input to RICA
                % params.lambda = 0.0005;   % sparsity cost
                params.lambda = 1e-2;
                params.numFeatures = 64; % number of filter banks to learn
                % params.epsilon = 1e-2;
                params.epsilon = 1e-6;
                params.m=20000; % num patches
                
                if(targetFlowerIndex == 1)
                    testData = [data(:,1:offset(1)) data(:,offset(4+targetTrafficIndex-1)+1:offset(4+targetTrafficIndex))];
                    testLabels = [ones(1,offset(1)) zeros(1,offset(4+targetTrafficIndex)-offset(4+targetTrafficIndex-1))] + 1;
                else
                    testData = [data(:,offset(targetFlowerIndex-1)+1:offset(targetFlowerIndex)) data(:,offset(4+targetTrafficIndex-1)+1:offset(4+targetTrafficIndex))];
                    testLabels = [ones(1,offset(targetFlowerIndex)-offset(targetFlowerIndex-1)) zeros(1,offset(4+targetTrafficIndex)-offset(4+targetTrafficIndex-1))] + 1;
                end;
                
                params.numM = size(trainData,1);                % input data feature dimensions
                params.numK = 10;   %80;%10;                                 % number of hidden units 
                params.numC = 2;
                params.alpha = 0.5;%1;
                params.beta = 0.5;%0.5; 
                params.gamma = 0.00001;%0.0001;
                params.lambda = 0.1;
                params.numX = numX;
                
                maniParameter.alpha = 2;
                maniParameter.lambda = 0.5;
                maniParameter.layers = 5;
                maniParameter.beta = 0;
                maniParameter.L = LaplacianMatrix(trainData,testData,10);
                
                unlabeledData = [trainData testData];
                
              %% ======================================================================
                %  STEP 2: Train the TLMRA
                xFinalRepresentation = TLMRA_Representation(trainData,testData,trainLabels,params,maniParameter);
                
                 xr=xFinalRepresentation(:,1:size(trainData,2));
                 xr=xr';
                 bestC = 1./mean(sum(xr.*xr,2));
                 model = svmtrain(trainLabels',xr,['-q -t 0 -c ',num2str(bestC),' -m 3000']);
                 xe = xFinalRepresentation(:,size(trainData,2)+1:size(unlabeledData,2));
                 xe=xe';
                [label,accuracy] = svmpredict(testLabels',xe,model);
                Result = [Result;accuracy(1)]
                
                iCnt = iCnt + 1;
            end
        end
    end
end
Result
xlswrite('C:\Users\pc\Documents\MATLAB\icbk\result\Amazon_Result.xlsx',Result,'data5');