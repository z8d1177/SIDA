function [ prevhx ] = ManifoldRegression(xx, parameters)

prevhx = xx;
allhx = [];
D_cell = cell(parameters.layers,1);
W_cell = cell(parameters.layers,1);
for layer = 1:parameters.layers
    disp([' layer:' num2str(layer) ]);
    [newhx,W, D] = ManiMethod_loss(prevhx, parameters);
    D_cell{layer} = D;
    W_cell{layer} = W;
    allhx = [allhx; newhx];
    prevhx = newhx;
    clear newhx;
end

end

