%run Improvedbwconncomp
tic
ResultPath='.\Data\';%分割结果路径
Dir=dir([ResultPath,'*.png']);
Images=zeros(2000,2000,31);
for j=1:length(Dir)
%     if  j==11
%     end
    filename=Dir(j).name;
    Result=imread([ResultPath filename]);
    Result=Result(1:2000,1:2000);%做小图 调试代码
    Images(:,:,j)=Result;
end

LabelMatrix=Improvedbwconncomp(Images,[0.01 0.4 0.5 0.03 1500]);
