clear;clc;close all
% %��mask_rcnnʶ�����������²�λ����Ϣɸѡ
% ��ǰ��Դ2�㣬ͬʱ����ɸѡ�븳�����ӹ�ϵ

class={'1','2'};
nameclass={'_1_1','_2_1'};
for k=1:2
    disp(k)
    zbPas = ['\\192.168.3.2\data01\users\hongb\T2\mask total2\total\',num2str(k),'\ɸѡǰlabel\'];
    dirZb = dir([zbPas,'*.png']);
    savePas = ['\\192.168.3.2\data01\users\hongb\T2\mask total2\total\',num2str(k),'\ɸѡ��label\'];
    mkdir(savePas);
    dis = 511;  % ���ĵ������ֵ��l1��ʽ��
    iou_threshold = 0.3;  % overlap��ֵ
    trace_layer = 2;   % ��Դ����,һ��2-3��
    trace_threshold = [0.3,0.3,0.2];  % ��Դÿһ����overlap��ֵ
    select_threshold =3; % ���ִ���С����ֵ���ģ�����ɾȥ
    count=0;
    tic
    temp_boxes=cell(size(dirZb,1),1);
    temp_connect=cell(size(dirZb,1),1);
    for i = 1 : size(dirZb,1)
        %         name = [num2str(i,'%03d'),'.png'];
        name = ['layer',num2str(i),nameclass{k},'.png'];
        im_label = imread(fullfile(zbPas,name));
        temp_boxes{i,1} = extract_bboxes(im_label);
%         nn=['D:\hongbei\Mask-rcnn\result\txt\',name];
%         temp_boxes{i,1}=load(nn);
        temp_connect{i,1} = [temp_boxes{i,1},-ones(size(temp_boxes{i,1},1),1)];
        
    end
    toc
    temp_connect_after=temp_connect;
    for pic = 1: size(dirZb,1)
        pic
        new_boxes = temp_boxes{pic,1};
        
        zb = new_boxes;
        if isempty(zb)
            continue;
        end
        zb_connect = temp_connect{pic,1};
        numZb = size(zb,1);
        
        %%%%%%%%%%new
        if pic == 1
            num_pass=size(zb,1);
            for num=1 : num_pass
                count=count+1;
                zb_connect(num,5)=count;
                temp_connect_after{pic,1}=zb_connect;
            end
        else
            if pic == 2
                zbU = temp_boxes{pic-1,1};  %����Դ�Ĳ���
                zbU_connect = temp_connect_after{pic-1,1};
                [new_zb_connect]=connect(zb,zbU,zb_connect,zbU_connect,trace_threshold(1));
                list=find(new_zb_connect(:,5)==-1);
                num_pass=size(list,1);
                for num=1 : num_pass
                    count=count+1;
                    new_zb_connect(list(num),5)=count;
                end
                
                temp_connect_after{pic,1}=new_zb_connect;
                
            else
                list_nor=cell(size(trace_layer,1),1);
                list=cell(size(trace_layer,1),1);
                for j = 1:trace_layer  %��Դ����
                    zbU = temp_boxes{pic-j,1};  %����Դ�Ĳ���
                    zbU_connect = temp_connect_after{pic-j,1};
                    if isempty(zbU) && j==1
                        list{j,1}=find(zb_connect(:,5)==-1);
                        list_nor{j,1}=find(zb_connect(:,5)~=-1); %У����ж�Ӧ������
                        num_pass=size(zb,1);
                        for num=1 : num_pass
                            count=count+1;
                            zb_connect(num,5)=count; 
                        end
                        temp_connect_after{pic,1}=zb_connect;
                        continue;
                    end
                    if isempty(zbU) && j~=1
                        continue;
                    end
                    [new_zb_connect]=connect(zb,zbU,zb_connect,zbU_connect,trace_threshold(j));
                    list{j,1}=find(new_zb_connect(:,5)==-1);
                    list_nor{j,1}=find(new_zb_connect(:,5)~=-1); %У����ж�Ӧ������
                    zb_validation=new_zb_connect(list{j,1},:);
                    %���¸�ֵ,У��ʣ��δƥ�����ǰ����Ĺ�ϵ
                    zb=zb_validation(:,1:4);
                    zb_connect=zb_validation;
                    if j==1
                        temp_connect_after{pic,1}=new_zb_connect;
                    else
                        if ~isempty(list_nor{j,1})
                            for i=1:size(list_nor{j,1},1)
                                t=list{1,1}(list_nor{j,1}(i));
                                temp_connect_after{pic,1}(t,:)=new_zb_connect(list_nor{j,1}(i),:);
                            end
                        end
                    end  % j==1
                    
                end  % j = 1:trace_layer
                list=find(temp_connect_after{pic,1}(:,5)==-1);
                num_pass=size(list,1);
                for num=1 : num_pass
                    count=count+1;
                    temp_connect_after{pic,1}(list(num),5)=count;
                end
            end
        end %if pic == 1
    end
    %%%%%%%%%%% ɸѡ���ڵĵ�
    tic
    txt_combine=[];
    num_connect_after=-ones(size(dirZb,1)+1,1);
    % �����ο����һ��������
    for ii=1:size(dirZb,1)
        num_connect_after(ii+1)=size(temp_connect_after{ii,1},1);
        txt_combine=[txt_combine;temp_connect_after{ii,1}];
    end
    toc
    % ��С����ֵ��label��Ϊ0
    for jj=1:count
        num_same=find(txt_combine(:,5)==jj);
        if size(num_same,1)<select_threshold
            txt_combine(num_same,5)=0;
        end
    end
    % ͳ�Ʋ�ͬ��label����
    unique(txt_combine(:,5));
    % imLabelɸ�������ٵ�label��������ͼƬ��ָ���ļ���
    select_connect = cell(size(dirZb,1),1);
    for iii=1:size(dirZb,1)
        a=max(0,sum(num_connect_after(2:iii)))+1;  %��ֹ��
        b=sum(num_connect_after(2:iii+1));   %��ֹ��
        select_connect{iii,1} = txt_combine(a:b,:);    %����iiiͼ��ľ��ο�
        list_txt_combine=txt_combine(a:b,5);
        idx=select_connect{iii,1} (all(list_txt_combine,2),:);
        %дͼ
        im=imread(fullfile(zbPas,['layer',num2str(iii),nameclass{k},'.png']));
        select_im=uint16(zeros(size(im)));
%         %����txt
%         fp=fopen(fullfile('D:\hongbei\Mask-rcnn\result\after txt\',[num2str(iii),'.txt']),'w');
%         [x_row,x_column]=size(idx);
%         for i_x=1:x_row
%             
%             fprintf(fp,'%d %d %d %d \r\n',idx(i_x,1:4));
%         end
%         txtxt{iii,1}=all(list_txt_combine,2)';
%         
%         fclose(fp);
        %%%%%%%%%%%%
        for l=1:size(idx,1)
            x=max(idx(l,1),1);
            y=idx(l,2);
            x1=idx(l,3);
            y1=idx(l,4);
            temp = im(y:y1,x:x1);
            imBw = im2bw(temp);                        %ת��Ϊ��ֵ��ͼ��
            imLabel = bwlabel(imBw);                %�Ը���ͨ����б��
            stats = regionprops(imLabel,'Area');    %�����ͨ��Ĵ�С
            area = cat(1,stats.Area);index = find(area == max(area));        %�������ͨ�������
            temp = ismember(imLabel,index);          %��ȡ�����ͨ��ͼ��
            select_im(y:y1,x:x1)=max(select_im(y:y1,x:x1),uint16(temp*idx(l,5)));
        end
%         figure;imshow(select_im,[])
        imwrite(select_im,[savePas,'layer',num2str(iii),nameclass{k},'.png'])
    end

end
% end %class






