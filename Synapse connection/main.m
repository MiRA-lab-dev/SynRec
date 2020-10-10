clear;clc;close all
% %对mask_rcnn识别结果根据上下层位置信息筛选
% 向前溯源2层，同时进行筛选与赋予连接关系

class={'1','2'};
nameclass={'_1_1','_2_1'};
for k=1:2
    disp(k)
    zbPas = ['\\192.168.3.2\data01\users\hongb\T2\mask total2\total\',num2str(k),'\筛选前label\'];
    dirZb = dir([zbPas,'*.png']);
    savePas = ['\\192.168.3.2\data01\users\hongb\T2\mask total2\total\',num2str(k),'\筛选后label\'];
    mkdir(savePas);
    dis = 511;  % 中心点距离阈值（l1范式）
    iou_threshold = 0.3;  % overlap阈值
    trace_layer = 2;   % 溯源层数,一般2-3层
    trace_threshold = [0.3,0.3,0.2];  % 溯源每一层框的overlap阈值
    select_threshold =3; % 出现次数小于阈值数的，将其删去
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
                zbU = temp_boxes{pic-1,1};  %被溯源的层数
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
                for j = 1:trace_layer  %溯源层数
                    zbU = temp_boxes{pic-j,1};  %被溯源的层数
                    zbU_connect = temp_connect_after{pic-j,1};
                    if isempty(zbU) && j==1
                        list{j,1}=find(zb_connect(:,5)==-1);
                        list_nor{j,1}=find(zb_connect(:,5)~=-1); %校验后有对应的行数
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
                    list_nor{j,1}=find(new_zb_connect(:,5)~=-1); %校验后有对应的行数
                    zb_validation=new_zb_connect(list{j,1},:);
                    %重新赋值,校验剩余未匹配框与前两层的关系
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
    %%%%%%%%%%% 筛选少于的点
    tic
    txt_combine=[];
    num_connect_after=-ones(size(dirZb,1)+1,1);
    % 将矩形框读入一个矩阵内
    for ii=1:size(dirZb,1)
        num_connect_after(ii+1)=size(temp_connect_after{ii,1},1);
        txt_combine=[txt_combine;temp_connect_after{ii,1}];
    end
    toc
    % 将小于阈值的label赋为0
    for jj=1:count
        num_same=find(txt_combine(:,5)==jj);
        if size(num_same,1)<select_threshold
            txt_combine(num_same,5)=0;
        end
    end
    % 统计不同的label个数
    unique(txt_combine(:,5));
    % imLabel筛除个数少的label，并保存图片到指定文件夹
    select_connect = cell(size(dirZb,1),1);
    for iii=1:size(dirZb,1)
        a=max(0,sum(num_connect_after(2:iii)))+1;  %起止点
        b=sum(num_connect_after(2:iii+1));   %终止点
        select_connect{iii,1} = txt_combine(a:b,:);    %属于iii图像的矩形框
        list_txt_combine=txt_combine(a:b,5);
        idx=select_connect{iii,1} (all(list_txt_combine,2),:);
        %写图
        im=imread(fullfile(zbPas,['layer',num2str(iii),nameclass{k},'.png']));
        select_im=uint16(zeros(size(im)));
%         %保存txt
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
            imBw = im2bw(temp);                        %转换为二值化图像
            imLabel = bwlabel(imBw);                %对各连通域进行标记
            stats = regionprops(imLabel,'Area');    %求各连通域的大小
            area = cat(1,stats.Area);index = find(area == max(area));        %求最大连通域的索引
            temp = ismember(imLabel,index);          %获取最大连通域图像
            select_im(y:y1,x:x1)=max(select_im(y:y1,x:x1),uint16(temp*idx(l,5)));
        end
%         figure;imshow(select_im,[])
        imwrite(select_im,[savePas,'layer',num2str(iii),nameclass{k},'.png'])
    end

end
% end %class






