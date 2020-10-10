%% 提取mask里面的矩形框，并将矩形框向外扩指定的长度
function new_boxes = extract_bboxes(im_label)

[row,column]=size(im_label);
im_label=(im_label>0.5);
% figure;imshow(im_label,[])
im_label=bwareaopen(im_label,50,4);
boxes=regionprops(im_label,'Boundingbox');
% img=bwlabel(img);
boxes=cat(1,boxes.BoundingBox);
if isempty(boxes)
    new_boxes=[];
else
    boxes(:,1)=round(boxes(:,1));
    boxes(:,2)=round(boxes(:,2));
    boxes(:,3)=round(boxes(:,1)+boxes(:,3));
    boxes(:,4)=round(boxes(:,2)+boxes(:,4));
    t=boxes(:,3)-boxes(:,1)>boxes(:,4)-boxes(:,2);
    temp1=boxes(find(t==1),:);
    temp2=boxes(find(t==0),:);
    
    a=(temp1(:,3) + 15 > column);
    temp1(a,3) = column;
    temp1(~a,3) = temp1(~a,3) + 15;
    
    b=(temp1(:,1) - 15 < 1);
    temp1(b,1) = 1;
    temp1(~b,1) = temp1(~b,1) - 15;
    
    c=(temp1(:,4) + 30 > row);
    temp1(c,4) = row;
    temp1(~c,4) = temp1(~c,4) + 30;
    
    d=(temp1(:,2) - 30 < 1);
    temp1(d,2) = 1;
    temp1(~d,2) = temp1(~d,2) - 30;
    
    e=(temp2(:,3) + 30 > column);
    temp2(e,3) = column;
    temp2(~e,3) = temp2(~e,3) + 30;
    
    f=(temp2(:,1) - 30 < 1);
    temp2(f,1) = 1;
    temp2(~f,1) = temp2(~f,1) - 30;
    
    g=(temp2(:,4) + 15 > row);
    temp2(g,4) = row;
    temp2(~g,4) = temp2(~g,4) + 15;
    
    h=(temp2(:,2) - 15 < 1);
    temp2(h,2) = 1;
    temp2(~h,2) = temp2(~h,2) - 15;
    
    new_boxes=[temp1;temp2];
end
end
