%% ���Ӵ���
% zb���������
% zbU���ϲ����
% zb_connect�������µ�һά�ı������
% zbU_connect�������µ�һά���ϲ����
% trace_threshold������֮��overlap��ֵ
% count������,��ʾ��ǰ��label��

function [new_zb_connect]=connect(zb,zbU,zb_connect,zbU_connect,trace_threshold)
    new_zb_connect=zb_connect;
    iou=boxoverlap(zb,zbU);
    % ɸ��С����ֵ�ķ���
    [max_value,Idx]=max(iou,[],2);
    max_value(find(max_value<trace_threshold))=0; %�ж����,����Ŀ��ֵ������ֵ��Ϊ0
    list_max_value=max_value~=0;  % Ŀ��list
    list_not_value=max_value==0;  % ��ɸ��list
    ind = find(list_not_value);
    Idx(~list_max_value)=0;
    %                 max_value=max_value(list_max_value,:);
    %                 Idx=Idx(list_max_value,:);
    % ��ֵlabel,���²�֮������ӹ�ϵ
    zb_leave=zb(list_max_value,:);
    zbU_corresponding=zbU_connect( Idx(Idx~=0),:);
    new_zb_connect(list_max_value,5)=zbU_corresponding(:,5);  %%���ܳ���bug��������ͬһ�ط���ֵ��һ��

end