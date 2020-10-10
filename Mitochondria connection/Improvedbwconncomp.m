function [IndexsM]=Improvedbwconncomp(varargin)
%Find connected components in binary image.
%LowThresold=0.01,HighThresold=0.4 are adopted to judge the coarse similarity 
%lambda=0.5 is a regularization parameter for balance of position term and  shape term; (0,1] are suggested for mitochondria
%Threshold2=0.03 is adopted to obtain the final connection matrices; (0,1] are suggested for mitochondria
%P=1500 is adopted ro remove the connected components (objects) that have fewer than P pixels 
Images=cell2mat(varargin(1));
if ismatrix(Images)
   error('The input should be a 3d matrix'); 
end
if nargin < 2
    LowThresold=0.01;
    HighThresold=0.2;% 0.4
    lambda=0.5;
    Threshold2=0.03;
    P=1500;
end
if nargin == 2
    Param=cell2mat(varargin(2));
    LowThresold=Param(1);
    HighThresold=Param(2);
    lambda=Param(3);
    Threshold2=Param(4);
    P=Param(5);
end
Length=size(Images,3);
for j=1:Length
    Result=Images(:,:,j);
%     if max(max(Result))>0
        %     Result=Result(1:4000,4001:end);%做小图 调试代码
        Coordinate=regionprops(Result>0,'BoundingBox');
        Coordinate=round(cell2mat((struct2cell(Coordinate))'));
        Coordinate(:,3)=Coordinate(:,3)+Coordinate(:,1);
        Coordinate(:,4)=Coordinate(:,4)+Coordinate(:,2);
        Coordinate=[Coordinate ones(size(Coordinate,1),1)];
%     else
%         Coordinate=[];
%     end
    Points(j,1)={Coordinate};
end
[ConnectionShip]=CoarseConnection(Points,LowThresold,HighThresold);
[ConnectionShip]=Validation(Images,ConnectionShip,Threshold2,lambda);
%%将每一个线粒体赋一个初始值
[ConnectionShip]=OverConnection(ConnectionShip);
%%根据分裂合并获取相同的连接关系
[ConnectionShip]=FineConnection(ConnectionShip);
%%考虑多层情况
[ConnectionShip]=Obtain_Omit_Segmentation(Images,ConnectionShip,Threshold2,lambda);
%%连接断层的点
[ConnectionShip]=Connect_Omit_Mito(ConnectionShip);
IndexsM=cell2mat(ConnectionShip.FinalConnection1);
Labels=unique(IndexsM(:,3));
IndexsM1=IndexsM;
for k=1:length(Labels)
    Label=Labels(k);
    Indk=find(IndexsM(:,3)==Label);
    IndexsM1(Indk,3)=k;
end 
IndexsM=IndexsM1;
% MaskN=zeros(size(Images));
% for i=1:Length
%    Indi=find(IndexsM(:,1)==i); 
%    TemVaule=IndexsM(Indi,2:3);
%    TemImage=Images(:,:,i);
%    [BW,Num]=bwlabel(TemImage);
%    Mask=zeros(size(TemImage));
%    for j=1:Num
%        Ind=find(BW==j);
%        Val=find(TemVaule(:,1)==j);
%        Mask(Ind)=TemVaule(Val,2);
%    end
%    MaskN(:,:,i)=Mask;
% end 
%    S = regionprops(MaskN, 'Area');
%    Indp=find([S.Area]>P);
%    MaskNN=zeros(size(Images));
%    for k=1:length(Indp)
%        Ind=Indp(k);
%        Indk=find(MaskN==Ind);
%        MaskNN(Indk)=k;
%    end
end
