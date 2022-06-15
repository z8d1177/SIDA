function M = MMD(src_X, tar_X, src_labels, tar_labels)
       probability= test_LR(src_X,tar_X,src_labels,tar_labels);
       probability  = probability'; 
       SourcePosLable = size(find(src_labels>0),1);
       SourceNegLable = size(src_labels,1)-SourcePosLable;
        
       TargetPosLable = size(find(probability'>0),1);
       TargetNegLable = size(probability',1)-TargetPosLable;
       
        Ms=[];
       for iters=1:size(src_labels,1)
           for itert=size(src_labels,1)
               if src_labels(iters)
                   if src_labels(itert)
                       Ms(iters,itert) = 1/SourcePosLable/SourcePosLable;
                   else
                       Ms(iters,itert) = -1/SourcePosLable/SourceNegLable;
                   end
               else
                   if src_labels(itert)
                       Ms(iters,itert) = -1/SourcePosLable/SourceNegLable;
                   else
                       Ms(iters,itert) = 1/SourceNegLable/SourceNegLable;
                   end 
               end
           end
       end
      
       Mst=[];
       for iters=1:size(src_labels,1)
           for itert=size(probability,1)
               if src_labels(iters)
                   if probability(itert)
                       Mst(iters,itert) = 1/SourcePosLable/TargetPosLable;
                   else
                       Mst(iters,itert) = -1/SourcePosLable/TargetNegLable;
                   end
               else
                   if probability(itert)
                       Mst(iters,itert) = -1/SourceNegLable/TargetPosLable;
                   else
                       Mst(iters,itert) = 1/SourceNegLable/TargetNegLable;
                   end 
               end
           end
       end
   
       Mt=[];
       for iters=1:size(probability,1)
           for itert=size(probability,1)
               if probability(iters)
                   if probability(itert)
                       Mt(iters,itert) = 1/TargetPosLable/TargetPosLable;
                   else
                       Mt(iters,itert) = -1/TargetPosLable/TargetNegLable;
                   end
               else
                   if probability(itert)
                       Mt(iters,itert) = -1/TargetNegLable/TargetPosLable;
                   else
                       Mt(iters,itert) = 1/TargetNegLable/TargetNegLable;
                   end 
               end
           end
       end
       M = [Ms,Mst;Mst',Mt];
       disp('M...');
end