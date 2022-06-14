%Clustering on COIL20
clear	
load('COIL20.mat');	
%nClass = length(unique(gnd));
nClass = size(fea,2);

%Normalize each data vector to have L2-norm equal to 1  
fea = NormalizeFea(fea);

%Clustering in the original space
rand('twister',5489);
label = litekmeans(fea,20,'Replicates',20);
MIhat = MutualInfo(gnd,label);
disp(['Clustering in the original space. MIhat: ',num2str(MIhat)]);
%Clustering in the original space. MIhat: 0.7386

tic;
nBasis = nClass;
SCCoptions.ReguParaType = 'SLEP';
SCCoptions.ReguGamma = 0.2;
[B, V] = SCC(fea', nBasis, SCCoptions); %'
CodingTime = toc;
disp(['Sparse Concept Coding with ',num2str(nBasis),' basis vectors. Time: ',num2str(CodingTime)]);
%Sparse Concept Coding with 20 basis vectors. Time: 1.924

rand('twister',5489);
label = litekmeans(V',nClass,'Replicates',20); %'
MIhat = MutualInfo(gnd,label);
disp(['Clustering in the ',num2str(nBasis),'-dim Sparse Concept Coding space, MIhat: ',num2str(MIhat)]);
%Clustering in the 20-dim Sparse Concept Coding space, MIhat: 0.90176

%stack
nClass = length(unique(gnd));
tic;
nBasis = nClass;
SCCoptions.ReguParaType = 'SLEP';
SCCoptions.ReguGamma = 0.2;
[B, V] = SCC(V, nBasis, SCCoptions); %'
CodingTime = toc;
disp(['Sparse Concept Coding with ',num2str(nBasis),' basis vectors. Time: ',num2str(CodingTime)]);
%Sparse Concept Coding with 20 basis vectors. Time: 1.924

rand('twister',5489);
label = litekmeans(V',nClass,'Replicates',20); %'
MIhat = MutualInfo(gnd,label);
disp(['Clustering in the ',num2str(nBasis),'-dim Sparse Concept Coding space, MIhat: ',num2str(MIhat)]);
%Clustering in the 20-dim Sparse Concept Coding space, MIhat: 0.90176


tic;
SCCoptions.ReguParaType = 'LARs';
SCCoptions.Cardi = 2:2:20;
[B, V, cardiCandi] = SCC(fea', nBasis, SCCoptions); %'
CodingTime = toc;
disp(['Sparse Concept Coding with ',num2str(nBasis),' basis vectors. Time: ',num2str(CodingTime)]);
%Sparse Concept Coding with 20 basis vectors. Time: 4.5091

for i = 1:length(cardiCandi)
  rand('twister',5489);
  label = litekmeans(V{i}',nClass,'Replicates',20); %'
  MIhat = MutualInfo(gnd,label);
  disp(['Clustering in the ',num2str(nBasis),'-dim Sparse Concept Coding space with cardinality ',num2str(cardiCandi(i)),'. MIhat: ',num2str(MIhat)]);
end
%Clustering in the 20-dim Sparse Concept Coding space with cardinality 2. MIhat: 0.68255
%Clustering in the 20-dim Sparse Concept Coding space with cardinality 4. MIhat: 0.79441
%Clustering in the 20-dim Sparse Concept Coding space with cardinality 6. MIhat: 0.84102
%Clustering in the 20-dim Sparse Concept Coding space with cardinality 8. MIhat: 0.81865
%Clustering in the 20-dim Sparse Concept Coding space with cardinality 10. MIhat: 0.84306
%Clustering in the 20-dim Sparse Concept Coding space with cardinality 12. MIhat: 0.88502
%Clustering in the 20-dim Sparse Concept Coding space with cardinality 14. MIhat: 0.88996
%Clustering in the 20-dim Sparse Concept Coding space with cardinality 16. MIhat: 0.88736
%Clustering in the 20-dim Sparse Concept Coding space with cardinality 18. MIhat: 0.87489
%Clustering in the 20-dim Sparse Concept Coding space with cardinality 20. MIhat: 0.88158