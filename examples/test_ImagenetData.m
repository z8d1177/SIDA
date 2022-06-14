%% ======================================================================
%  STEP 0: parameters

addpath('C:\Users\pc\Documents\MATLAB\unbalance\method\')
addpath('C:\Users\pc\Documents\MATLAB\unbalance\examples\libsvm-3.11\matlab\');

global params;
global maniParameter;


% global numX;               % number of positive and negative instances in traindata

%% ======================================================================
%  STEP 1: Load data
load('C:\Users\pc\Documents\MATLAB\unbalance\dataset\imageNet(feature=900).mat');
for sourceIndex = 1:4
    for targetIndex = 1:4
        if sourceIndex == targetIndex
            continue
        end
        
       if(sourceIndex == 1)
            images_train = [ambulancedata scooterdata];
            labels_train = [ones(1,size(ambulancedata,2)) zeros(1,size(scooterdata,2))];
            numX = [size(ambulancedata,2) size(scooterdata,2)];
        end;
        if(sourceIndex == 2)
            images_train = [taxidata scooterdata];
            labels_train = [ones(1,size(taxidata,2)) zeros(1,size(scooterdata,2))];
            numX = [size(taxidata,2) size(scooterdata,2)];
        end;
        if(sourceIndex == 3)
            images_train = [jeepdata scooterdata];
            labels_train = [ones(1,size(jeepdata,2)) zeros(1,size(scooterdata,2))];
            numX = [size(jeepdata,2) size(scooterdata,2)];
        end;
        if(sourceIndex == 4)
            images_train = [minivandata scooterdata];
            labels_train = [ones(1,size(minivandata,2)) zeros(1,size(scooterdata,2))];
            numX = [size(minivandata,2) size(scooterdata,2)];
        end;
            
        if(targetIndex == 1)
            images_Test = [ambulancedata scooterdata];
            labels_Test = [ones(1,size(ambulancedata,2)) zeros(1,size(scooterdata,2))]; 
        end;
        if(targetIndex == 2)
            images_Test = [taxidata scooterdata];
            labels_Test = [ones(1,size(taxidata,2)) zeros(1,size(scooterdata,2))];
        end;
        if(targetIndex == 3)
            images_Test = [jeepdata scooterdata];
            labels_Test = [ones(1,size(jeepdata,2)) zeros(1,size(scooterdata,2))];
        end;
        if(targetIndex == 4)
            images_Test = [minivandata scooterdata];
            labels_Test = [ones(1,size(minivandata,2)) zeros(1,size(scooterdata,2))];
       end;
%        labels_train = labels_train + 1;
%        labels_Test = labels_Test + 1;
       
      %% ======================================================================
       %%STEP 2: Initialize the parameter by Sparse SDA
%        [images_train, labels_train] = load_MNIST_data('C:\Users\pc\Documents\MATLAB\unbalance\dataset\train-images.idx3-ubyte',...
%             'C:\Users\pc\Documents\MATLAB\unbalance\dataset\train-labels.idx1-ubyte', 'min_max_scaler', 0);
%         [images_Test, labels_Test] = load_MNIST_data('C:\Users\pc\Documents\MATLAB\unbalance\dataset\t10k-images.idx3-ubyte',...
%             'C:\Users\pc\Documents\MATLAB\unbalance\dataset\t10k-labels.idx1-ubyte', 'min_max_scaler', 0);

%         theta = initialize_Sparse_SDA(images_train, labels_train', images_Test, labels_Test');    % Randomly initialize the parameters
          theta = UDDSA(images_train, labels_train', images_Test, labels_Test', numX);    % Randomly initialize the parameters
       
    end
end