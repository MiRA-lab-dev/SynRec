%% 连接代码
% zb：本层检测框
% zbU：上层检测框
% zb_connect：加上新的一维的本层检测框
% zbU_connect：加上新的一维的上层检测框
% trace_threshold：检测框之间overlap阈值
% count：计数,表示当前的label数

function [new_zb_connect]=connect(zb,zbU,zb_connect,zbU_connect,trace_threshold)
    new_zb_connect=zb_connect;
    iou=boxoverlap(zb,zbU);
    % 筛出小于阈值的方框
    [max_value,Idx]=max(iou,[],2);
    max_value(find(max_value<trace_threshold))=0; %判断语句,保留目标值，其余值赋为0
    list_max_value=max_value~=0;  % 目标list
    list_not_value=max_value==0;  % 被筛的list
    ind = find(list_not_value);
    Idx(~list_max_value)=0;
    %                 max_value=max_value(list_max_value,:);
    %                 Idx=Idx(list_max_value,:);
    % 赋值label,上下层之间的连接关系
    zb_leave=zb(list_max_value,:);
    zbU_corresponding=zbU_connect( Idx(Idx~=0),:);
    new_zb_connect(list_max_value,5)=zbU_corresponding(:,5);  %%可能出现bug？？比如同一地方赋值不一样

end