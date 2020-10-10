clear all;
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
files_no = 31;
iters_no = floor(files_no/10);
count = 1;
IndexsMs=cell(iters_no,1);
n = 11; % 每一块的个数
for j=1:iters_no
    if j==iters_no
        n=files_no-(iters_no-1)*10;
    end
masks = zeros(8416,8624,n);
% masks1 = zeros(20000,32000,n);
tic
for i =1:n
    tt = (j-1)*10+i
%     imName = [num2str(i,'%d'),'_pred','.png'];
    imName = ['./data/',num2str(tt,'%03d'),'.png'];
    im = imread(imName);
    im = im>127;
%     se = strel('disk',1);
%     im = imdilate(im,se);
    im = bwareaopen(im,500);
%     im1 = imresize(im,[10000,16000]);
%     [BWs,NumS]=bwlabel(im1);
    [BW, Num]=bwlabel(im);
    if Num==0
        bug = 1;
%         imName = ['\\192.168.3.15\publicShare\data\flyLobe\mito_sparse\',num2str((j-1)*10+i+1273,'%04d'),'.png'];
%         im = imread(imName);
    end
    masks(:,:,i) = im;
%     masks1(:,:,1) = im;
%     if NumS~= Num
%         bug(count) = (j-1)*10+i+500;
%         count=count+1;
%     end
end
IndexsM = Improvedbwconncomp(masks);
IndexsMs(j,1)={IndexsM};
% 
toc
end
% save('IndexsMs_E62_1_2.mat','IndexsMs')

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clear all;

IndexsM1 = cell2mat(IndexsMs(1,1));
IndexsM1(:,1) = IndexsM1(:,1); 
%%
for j=2:iters_no
    IndexsM2 = cell2mat(IndexsMs(j,1));
    IndexMax = max(IndexsM1(:,3));
    IndexsM2(:,1) = IndexsM2(:,1) + (j-1)*10;
    IndexsM2(:,3) = IndexsM2(:,3) + IndexMax; %% 加上最大的label 使得前后的标签不一样
    MaxLayer = max(IndexsM1(:,1));
    IndMax = find(IndexsM1(:,1)==MaxLayer);
    IndexMaxLayer = IndexsM1(IndMax,:);%重叠的层上
    
    IndMin = find(IndexsM2(:,1)==(j-1)*10+1);
    IndexMinLayer = IndexsM2(IndMin,:);%重叠的层下
    
%     Matrix = zeros(size(IndexMinLayer,1),3);
%     for i =1:size(Matrix,1)
%         Matrix(i,1)=i;
%         Indi = find(IndexMaxLayer(:,2)==i);
%         Matrix(i,2)=IndexMaxLayer(Indi,3);
%         Indi = find(IndexMinLayer(:,2)==i);
%         Matrix(i,3)=IndexMinLayer(Indi,3);
%     end
%     
%     S = tabulate(Matrix(:,2));
%     S(S(:,3)==0,:)=[];
%     M = tabulate(Matrix(:,3));
%     M(M(:,3)==0,:)=[];
    mito_list = IndexMinLayer(:,2);
    i = 1;
    while  size(IndexMinLayer,1)>=1 %对第mito个线粒体进行处理
        mito = mito_list(i);
        Indi1 = find(IndexMaxLayer(:,2)==mito);
        TemLabel= IndexMaxLayer(Indi1,3);
        Indi2 = find(IndexMinLayer(:,2)==mito);
        NextLabel = IndexMinLayer(Indi2,3);
        %%%分三种情况
        merge = find(IndexMinLayer(:,3)==NextLabel);
        if size(merge,1)>1
            jjj = 1;
            indexs_merge =  [];
           indexs_max = [];
           indexs_mito = [];
            for s = 1: size(merge,1)
                num = IndexMinLayer(merge(s),2);%第num个线粒体
                label_In_Max = IndexMaxLayer(find(IndexMaxLayer(:,2)==num),3);
                nums_correction = IndexMaxLayer(find(IndexMaxLayer(:,3)==label_In_Max),2);
%                 label_In_Max = Matrix(merge(s),2);
                IndN = find(IndexsM1(:,3)==label_In_Max);
                IndexsM1(IndN,3) = NextLabel;
                for jj =1:size(nums_correction,1)
                    indexs_merge(jjj) = find(IndexMinLayer(:,2)==nums_correction(jj));
                    indexs_max(jjj) = find(IndexMaxLayer(:,2)==nums_correction(jj));
                    indexs_mito(jjj) = find(mito_list==nums_correction(jj));
                    
                    label_In_Min = IndexMinLayer(indexs_merge(jjj),3);
                    if label_In_Min ~= NextLabel
                        IndN = find(IndexsM2(:,3)==label_In_Min);
                        IndexsM2(IndN,3) =NextLabel;
                    end
                    jjj = jjj+1;
%                     IndexMaxLayer(find(IndexMaxLayer(:,2)==nums_correction(jj)),:)=[];
%                     mito_list(find(mito_list==nums_correction(jj)))=[];
                end
            end
           indexs_merge =  unique(indexs_merge);
           indexs_max = unique(indexs_max);
           indexs_mito = unique(indexs_mito);
           IndexMinLayer(indexs_merge,:)=[];
           IndexMaxLayer(indexs_max,:)=[];
           mito_list(indexs_mito,:)=[];
%             i = i + size(merge,1);
        end
        split = find(IndexMaxLayer(:,3)==TemLabel);
        if size(split,1)>1
            iii = 1;
            indexs_split = [];
           indexs_min = [];
           indexs_mito = [];
            for t =1: size(split,1)
                num= IndexMaxLayer(split(t),2);
                label_In_Min = IndexMinLayer(find(IndexMinLayer(:,2)==num),3);
%                 label_In_Max = Matrix(merge(s),2);
                nums_correction = IndexMinLayer(find(IndexMinLayer(:,3)==label_In_Min),2);
                IndN = find(IndexsM2(:,3)==label_In_Min);
                IndexsM2(IndN,3) = TemLabel;
                for ii = 1:size(nums_correction,1)
                    indexs_split(iii) = find(IndexMaxLayer(:,2)==nums_correction(ii));
                    indexs_min(iii) = find(IndexMinLayer(:,2)==nums_correction(ii));
                    indexs_mito(iii) = find(mito_list==nums_correction(ii));
                    
                    label_In_Max = IndexMaxLayer(indexs_split(iii),3);
                    if label_In_Max ~= TemLabel
                        IndN = find(IndexsM1(:,3)==label_In_Max);
                        IndexsM1(IndN,3) = TemLabel;
                    end
                    iii = iii + 1;
%                     IndexMinLayer(find(IndexMinLayer(:,2)==nums_correction(ii)),:) = [];
%                     mito_list(find(mito_list==nums_correction(ii)))=[];
                end
            end
            indexs_split =  unique(indexs_split);
           indexs_min = unique(indexs_min);
           indexs_mito = unique(indexs_mito);
            IndexMaxLayer(indexs_split,:)=[];
            IndexMinLayer(indexs_min,:)=[];
            mito_list(indexs_mito,:)=[];
%             i = i + size(split,1);
        end

        
        
        if size(merge,1)==1 && size(split,1)==1
            IndN =find(IndexsM2(:,3)==NextLabel);
            IndexsM2(IndN,3) =  TemLabel;
            %%
            IndexMaxLayer(Indi1,:)=[];
            IndexMinLayer(Indi2,:) = [];
            mito_list(find(mito_list==mito))=[];
%             i = i+1;
        end
        
    end
    IndexsM1 = [IndexsM1; IndexsM2];
    IndexsM1 = unique(IndexsM1,'rows');
end


%% label 可能不连续  以下操作将label从一开始排序(可选)
Labels=unique(IndexsM1(:,3));
IndexsM2=IndexsM1;
for k=1:length(Labels)
    Label=Labels(k);
    Indk=find(IndexsM2(:,3)==Label);
    IndexsM2(Indk,3)=k;
end 
IndexsM=IndexsM2;


%%%%%%%%%%%%%%%%%%
for i = 1:iters_no
    nnn = (i-1)*10+1;
    Indi=find(IndexsM(:,1) == nnn);
    aa = IndexsM(Indi,2);
    if size(aa,1)~= max(aa)
        bug = 1
    end
end
%%%%%%%%%%%%%%%%%%


for i = 1:501
%     i+1273
%     imName = [num2str(i,'%d'),'_pred','.png'];
    imName = ['./data/',num2str(i,'%03d'),'.png'];
    im = imread(imName);
    im = im>127;
%     se = strel('disk',1);
%     im = imdilate(im,se);
    im = bwareaopen(im,500);
%     im1 = imresize(im,[10000,16000]);
  
   Indi=find(IndexsM(:,1)==i); 
   TemVaule=IndexsM(Indi,2:3);
   TemImage=im;
   [BW,Num]=bwlabel(TemImage);
   Mask=zeros(size(TemImage));
   for j=1:Num
       Ind=find(BW==j);
       Val=find(TemVaule(:,1)==j);
       Mask(Ind)=TemVaule(Val,2);
   end
   imName = ['./savepath/',num2str(i),'.png'];
%    Mask = imresize(Mask,[24000,36000],'nearest');
   imwrite(uint16(Mask),imName);
end

