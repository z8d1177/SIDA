function theta_SAE = train_UDDSA(images_Train, labels_Train, images_Test, labels_Test,architecture, option, option_unbalance)
%训练UDDSA

option_AE = option.option_AE; % 得到AE的一些预选参数
option_BP = option.option_BP; % 得到BP的一些预选参数

% 初始化网络参数 theta4SAE：用于存储堆叠起来的网络的参数
if strcmp(option_BP.activation, 'softmax') % softmax那一层不用偏置b
    count_W = architecture * [architecture(2:end) 0]';
    count_B = sum(architecture(2:(end - 1)));
    theta_SAE = zeros(count_W + count_B, 1);
else
    count_W = architecture * [architecture(2:end) 0]';
    count_B = sum(architecture(2:end));
    theta_SAE = zeros(count_W + count_B, 1);
end

inputdata = [images_Train images_Test];

%% 首先训练多个AE：按 architecture 训练
start_index = 1; % 存储变量的下标起点
for count_AE = 1 : (length(architecture) - 2) % 最后两层用于BP训练
    % AE网络的结构: inputSize -> hiddenSize -> outputSize
    architecture_AE = ...
        [architecture(count_AE) ...
        architecture(count_AE + 1) ...
        architecture(count_AE)];
    theta_AE  = init_parameters(architecture_AE); % 依据网络结构初始化网络参数
    
    [opt_theta, cost] = train_AE(inputdata, theta_AE, architecture_AE, count_AE, option_AE);
%     if count_AE == 1 % 可以根据cost的情况，判断是否还需要继续训练
%         [ opt_theta, cost ] = train_AE( input, opt_theta, architecture_AE, option_AE );
%     end
    
    disp(['第' num2str(count_AE) '层AE "' ...
        num2str(architecture_AE) '" 的训练误差是：'...
        num2str(cost)]);
    
    % 存储 AE的W1，b1 到 SAE 中
    end_index = architecture(count_AE) * architecture(count_AE + 1) + ...
        architecture(count_AE + 1) + start_index - 1;% 存储变量的下标终点
    theta_SAE(start_index : end_index) = opt_theta(1 : ...
        (architecture(count_AE) * architecture(count_AE + 1) + architecture(count_AE + 1)));
    
    % 修改input为上一层的output
    clear predict theta_AE opt_theta cost
    predict = predict_NN(inputdata, architecture_AE(1:2),...
        theta_SAE( start_index : end_index ), option_AE);
    inputdata = predict;
    
    start_index = end_index + 1;
end


%% 其次训练非平衡的Loss function并迭代优化参数矩阵
loss_options.Method = 'lbfgs'; %L-BFGS to optimize our cost function.
loss_options.maxIter = 300;	  % Maximum number of iterations of L-BFGS to run
loss_options.display = 'on';
loss_options.TolFun  = 1e-6;
loss_options.TolX = 1e-1119;
loss_options.maxFunEvals = 4000;
label1 = [labels_Train';ones(1,size(labels_Train,1))-labels_Train'];
[ub_opttheta, ub_cost] = minFunc( @(p) computeObjectAndGradiendForUnbalance(p, option_unbalance, images_Train, images_Test, label1), theta_SAE, loss_options);  

%% BP：训练最后两层
architecture_BP = [architecture(end-1) architecture(end)]; % 设置 BP 网络结构
% 依据网络结构初始化 BP网络参数
if strcmp(option_BP.activation, 'softmax') % softmax那一层不用偏置b
    last_active_is_softmax = 1;
    theta_BP = init_parameters(architecture_BP, last_active_is_softmax);
else
    theta_BP = init_parameters(architecture_BP);
end

[opt_theta, cost] = train_BPNN(input, output, theta_BP, architecture_BP, option_BP); % 训练BP网络
disp(['最后一层BP "' num2str(architecture_BP) '" 的训练误差是：' num2str(cost)]);

theta_SAE(start_index : end) = opt_theta;
    
end