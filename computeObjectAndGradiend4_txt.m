function [object,grad] = computeObjectAndGradiend4_txt(theta, numM, numK, numC, numX, alpha, traindata, testdata, trainlabel)

    % convert theta to the (W1 W2 W11 W22 b1 b2 b11 b22) matrix/vector format
	W1 = reshape(theta(1:numK*numM), numK, numM);
    W2 = reshape(theta(numK*numM+1:numK*numM+numK*numC), numC, numK);
    W22 = reshape(theta(numK*numM+numK*numC+1:numK*numM+2*numK*numC), numK, numC);
    W11 = reshape(theta(numK*numM+2*numK*numC+1:2*numK*numM+2*numK*numC), numM, numK);
    b1 = theta(2*numK*numM+2*numK*numC+1:2*numK*numM+2*numK*numC+numK);
	b2 = theta(2*numK*numM+2*numK*numC+numK+1:2*numK*numM+2*numK*numC+numK+numC);
	b22 = theta(2*numK*numM+2*numK*numC+numK+numC+1:2*numK*numM+2*numK*numC+2*numK+numC);
	b11 = theta(2*numK*numM+2*numK*numC+2*numK+numC+1:2*numK*numM+2*numK*numC+2*numK+numC+numM);
    data = [traindata testdata];
    % Cost and gradient
    datasize = size(data, 2);   
    weightsbuffer = ones(1, datasize);
    hiddeninputs = W1 * data + b1 * weightsbuffer;   % numK * datasize
    hiddenvalues = sigmoid(hiddeninputs);   % numK * datasize
    labelinputs = W2 * hiddenvalues + b2 * weightsbuffer;   % numC * datasize
    labelvalues = sigmoid(labelinputs);   % numC * datasize
    hiddeninputs1 = W22 * labelvalues + b22 * weightsbuffer;   % numK * datasize
    hiddenvalues1 = sigmoid(hiddeninputs1);   % numK * datasize

    finalinputs = W11 * hiddenvalues1 + b11 * weightsbuffer; % numM * datasize
    outputs = sigmoid(finalinputs); % numM * datasize
    errors = outputs - data; %visiblesize * numpatches
    clear hiddeninputs  hiddeninputs1 finalinputs
    % Calculate J1  求解第一项
    J1 = sum(sum((errors .* errors)))/datasize;
    fprintf('J1: %d\n',J1);
    
    % Calculate J2  %求解第二项
    traindatasize = size(traindata, 2);
    testsize = datasize - traindatasize;
    Ps = sum(hiddenvalues(:,1:1:traindatasize), 2)./traindatasize;
    Ps = Ps/sum(Ps);
    Pt = sum(hiddenvalues(:,traindatasize+1:1:datasize), 2)./testsize;
    Pt = Pt/sum(Pt);
    J2 = sum(Ps.*log(Ps./Pt)) + sum(Pt.*log(Pt./Ps));
    fprintf('J2: %d\n',J2);
    
    % Calculate Object
    object = J1 + alpha * J2;
    fprintf('object: %d\n',object);   
    
    clear J1 J2 J3 J4;
    
    AA = errors.*outputs.*(1-outputs);
    BB = hiddenvalues1.*(1-hiddenvalues1);
    CC = labelvalues.*(1-labelvalues);
    DD = hiddenvalues.*(1-hiddenvalues);
    %计算W1 b1梯度
    W1grad1 = zeros(numK,numM);
    W1grad2 = zeros(numK,numM);
%     W1grad3 = zeros(numK,numM);
    b1grad1 = zeros(numK,1);
    b1grad2 = zeros(numK,1);
%     b1grad3 = zeros(numK,1);
    W1grad1 = W1grad1 + 2*W2'*(W22'*(W11'*AA.*BB).*CC).*DD*data'/datasize;
    b1grad1 = b1grad1 + 2*W2'*(W22'*(W11'*AA.*BB).*CC).*DD*ones(datasize,1)/datasize;
%     fprintf('W1grad1: %d\n',W1grad1);   
%     fprintf('b1grad1: %d\n',b1grad1); 

%J2 = sum(Ps.*log(Ps./Pt)) + sum(Pt.*log(Pt./Ps)); 
%注：W1grad2第一行算式中有一项，即-Pt./Ps这一项，是在求sum(Pt.*log(Pt./Ps))中合并过来的，即J2的第二部分
%同理，W1grad2第二样算式中的-Ps./Pt这一项，是在求J2的第一部分中合并过来的，这部分要用纸和笔手动计算一下，比较复杂
    W1grad2 = W1grad2 + DD(:,1:1:traindatasize).*((1-Pt./Ps+log(Ps./Pt))*ones(1,traindatasize))*data(:,1:1:traindatasize)'/traindatasize;
    W1grad2 = W1grad2 + DD(:,traindatasize+1:1:datasize).*((1-Ps./Pt+log(Pt./Ps))*ones(1,testsize))*data(:,traindatasize+1:1:datasize)'/testsize;
    b1grad2 = b1grad2 + DD(:,1:1:traindatasize)*ones(traindatasize,1).*(1-Pt./Ps+log(Ps./Pt))/traindatasize;
    b1grad2 = b1grad2 + DD(:,traindatasize+1:1:datasize)*ones(testsize,1).*(1-Ps./Pt+log(Pt./Ps))/testsize;
%     W1grad3 = W1grad3 + 2*W2'*((labelvalues(:,1:1:traindatasize)-[trainlabel;ones(1,traindatasize)-trainlabel]).*CC(:,1:1:traindatasize)).*DD(:,1:1:traindatasize)*data(:,1:1:traindatasize)';
%     b1grad3 = b1grad3 + 2*W2'*((labelvalues(:,1:1:traindatasize)-[trainlabel;ones(1,traindatasize)-trainlabel]).*CC(:,1:1:traindatasize)).*DD(:,1:1:traindatasize)*ones(traindatasize,1);
    W1grad = W1grad1 + alpha * W1grad2;   
    b1grad = b1grad1 + alpha * b1grad2; 
    clear W1grad1 W1grad2 W1grad3 b1grad1 b1grad2 b1grad3;
    
    %计算W2 b2梯度
    W2grad1 = zeros(numC,numK);
%     W2grad2 = zeros(numC,numK);
    b2grad1 = zeros(numC,1);
%     b2grad2 = zeros(numC,1);
    W2grad1 = W2grad1 + 2*W22'*(W11'*AA.*BB).*CC*hiddenvalues'/datasize;
    b2grad1 = b2grad1 + 2*W22'*(W11'*AA.*BB).*CC*ones(datasize,1)/datasize;
%     W2grad2 = W2grad2 + 2*(labelvalues(:,1:1:traindatasize)-[trainlabel;ones(1,traindatasize)-trainlabel]).*CC(:,1:1:traindatasize)*hiddenvalues(:,1:1:traindatasize)';
%     b2grad2 = b2grad2 + 2*(labelvalues(:,1:1:traindatasize)-[trainlabel;ones(1,traindatasize)-trainlabel]).*CC(:,1:1:traindatasize)*ones(traindatasize,1);
    W2grad = W2grad1;   
    b2grad = b2grad1; 
    clear W2grad1 W2grad2 b2grad1 b2grad2;
    
    %计算W22 b22梯度
    W22grad1 = zeros(numK,numC);
    b22grad1 = zeros(numK,1);
    W22grad1 = W22grad1 + 2*W11'*AA.*BB*labelvalues'/datasize;
    b22grad1 = b22grad1 + 2*W11'*AA.*BB*ones(datasize,1)/datasize;
    W22grad = W22grad1;   
    b22grad = b22grad1; 
    clear W22grad1 b22grad1;
    
    %计算W11 b11梯度
    W11grad1 = zeros(numM,numK);
    b11grad1 = zeros(numM,1);
    W11grad1 = W11grad1 + 2*AA*hiddenvalues1'/datasize;
    b11grad1 = b11grad1 + 2*AA*ones(datasize,1)/datasize;
    W11grad = W11grad1;   
    b11grad = b11grad1; 
    clear W11grad1 b11grad1;
    
    grad = [W1grad(:) ; W2grad(:) ; W22grad(:) ; W11grad(:) ; b1grad(:) ; b2grad(:) ; b22grad(:) ; b11grad(:)];
    clear W1grad W2grad b1grad b2grad W11grad W22grad b11grad b22grad hiddenvalues hiddenvalues1 labelvalues labelinputs errors outputs data;
end

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end