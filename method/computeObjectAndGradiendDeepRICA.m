function [object,grad] = computeObjectAndGradiendDeepRICA(theta, numM, numK, numC, numX, alpha, beta, gamma, params,traindata, testdata, trainlabel)

    % convert theta to the (W1 W2 W11 W22 b1 b2 b11 b22) matrix/vector format
	W1 = reshape(theta(1:numK*numM), numK, numM);
    W2 = reshape(theta(numK*numM+1:numK*numM+numC*numK), numC, numK);
   
    data = [traindata testdata];
    % Cost and gradient
    datasize = size(data, 2);   
    
%     hiddenvalues = W1*data;
%     outputs = W2*hiddenvalues;
    hiddenvalues = params.lambda * sqrt((W1*data).^2 + params.epsilon);
    outputs = params.lambda * sqrt((W2*hiddenvalues).^2 + params.epsilon);
   % errors = outputs - data; %visiblesize * numpatches
    
    % Calculate J1  求解第一项
%     J1 = sum(sum((errors .* errors)))/datasize;
%     fprintf('J1: %d\n',J1);
    
    % Calculate J2  %求解第二项
    traindatasize = size(traindata, 2);
    testdatasize = datasize - traindatasize;
    Ps = sum(hiddenvalues(:,1:1:traindatasize), 2)./traindatasize;
    Ps = Ps/sum(Ps);
    Pt = sum(hiddenvalues(:,traindatasize+1:1:datasize), 2)./testdatasize;
    Pt = Pt/sum(Pt);
    J2 = sum(Ps.*log(Ps./Pt)) + sum(Pt.*log(Pt./Ps));
    fprintf('J2: %d\n',J2);
    
    % Calculate J3 %求解第三项
    J3=0;
    total = sum(exp(outputs(:,1:1:traindatasize)));
    ee = exp(params.lambda * sqrt((W2(1,:)*hiddenvalues(:,1:1:numX(1,1))).^2 + params.epsilon));
    J3 = J3 + sum(log(ee./total(:,1:1:numX(1,1))+eps));
    ee = exp(params.lambda * sqrt((W2(2,:)*hiddenvalues(:,numX(1,1)+1:1:numX(1,1)+numX(1,2))).^2 + params.epsilon));                       
    J3 = J3 + sum(log(ee./total(:,numX(1,1)+1:1:numX(1,1)+numX(1,2))));
    J3 = J3/traindatasize;
    fprintf('J3: %d\n',J3);   
    
    % Calculate J4
    J4 = sum(sum(W1 .* W1)) +sum(sum(W2 .* W2));
    fprintf('J4: %d\n',J4);
    
    % Calculate Object
%     object = J1 + alpha * J2 - beta * J3 + gamma * J4;
    object = alpha * J2 - beta * J3 + gamma * J4;
    fprintf('object: %d\n',object);   
    
    clear J2 J3 J4;
    
    AA = params.lambda * ((W2*hiddenvalues) ./ sqrt((W2*hiddenvalues).^2 + params.epsilon));
    BB = params.lambda * ((W1*data) ./ sqrt((W1*data).^2 + params.epsilon));
     
    %计算W1梯度
%     W1grad1 = zeros(numK,numM);
%     W1grad1 = W1grad1 + 2*errors*(params.lambda * ((W2*hiddenvalues) ./sqrt((W2*hiddenvalues).^2 + params.epsilon)) * W2)*(params.lambda * ((W1*data) ./sqrt((W1*data).^2 + params.epsilon)) * data')  / datasize;    
    W1grad2 = zeros(numK,numM);
    
    ptvalue = 1-Pt./Ps+real(log(Ps./Pt));
    ptvalue = ptvalue*ones(1,traindatasize);
    ptvalue = ptvalue*data(:,1:1:traindatasize)';
    W1grad2 = W1grad2 + (BB*data').*ptvalue/traindatasize;
    
    psvalue = 1-Ps./Pt+real(log(Pt./Ps));
    psvalue = psvalue*ones(1,testdatasize);
    psvalue = psvalue * data(:,traindatasize+1:1:datasize)';
    W1grad2 = W1grad2 + (BB*data').*psvalue/testdatasize;
%     W1grad2 = W1grad2 + BB.*((1-Ps./Pt+real(log(Pt./Ps)))*ones(1,testdatasize))*data(:,traindatasize+1:1:datasize)'/testdatasize;

    W1grad3 = zeros(numK,numM);
    W2_find = [W2(1,:)'*ones(1,numX(1,1)) W2(2,:)'*ones(1,numX(1,2))];
    ee = exp(outputs(:,1:1:traindatasize))+eps;
%     temp0 = W2_find - W2'*ee./(ones(numK,1)*total(1,:));
%     tempAA = AA(:,1:1:traindatasize);
%     temp1 = temp0 .* tempAA';
%     temp = temp1 .* BB(:,1:1:traindatasize)';
    temp = (W2_find - W2'*ee./(ones(numK,1)*total(1,:))) .* (W2' * AA(:,1:1:traindatasize)) .* BB(:,1:1:traindatasize);
    W1grad3 = W1grad3 + temp*data(:,1:1:traindatasize)';
    W1grad3 = W1grad3/traindatasize;
    
    W1grad = alpha * W1grad2 - beta * W1grad3 +  2 * gamma * W1;      
   
    clear W1grad2 W1grad3 ;
    
    %计算W2梯度
    W2grad3 = zeros(numC,numK);  
    total = sum(ee)+eps;
%     W2grad3(1,:) = (hiddenvalues(:,1:1:numX(1,1))*ones(numX(1,1),1))';
%     W2grad3(2,:) = (hiddenvalues(:,numX(1,1)+1:1:sum(numX))*ones(numX(1,2),1))';
    W2grad3 = AA(:,1:1:traindatasize) * hiddenvalues(:,1:1:traindatasize)';
    
    w2Temp0 = ee./(ones(numC,1)*total(1,:));
    w2Temp1 = (w2Temp0 .* AA(:,1:1:traindatasize)) * hiddenvalues(:,1:1:traindatasize)';
    W2grad3 = W2grad3 - w2Temp1;
    %     W2grad3 = (W2grad3 - ee./(ones(numC,1)*total(1,:))*hiddenvalues(:,1:1:traindatasize)') .* AA;
    W2grad3 = W2grad3/traindatasize;
    
    W2grad =  - beta * W2grad3 +  2 * gamma * W2;   
   
    clear W2grad3 ;
    
    grad = [W1grad(:) ; W2grad(:)];
    clear W1grad W2grad hiddenvalues outputs data;
end