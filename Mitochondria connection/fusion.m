function[cc,labels_list] = fusion(IndexMaxLayer,IndexMinLayer)
labels = unique(IndexMaxLayer(:,3));

for i = 1:size(labels,1)
    l = labels(i);
    seq_no_Max = IndexMaxLayer(IndexMaxLayer(:,3)==l,2);
    todo = seq_no_Max;
    seq_no = seq_no_Max;
    labels_list = [l];
    while 1
        currents = todo;
        for j = 1:size(currents,1)
            current = currents(j);
            label_Min = IndexMinLayer(IndexMinLayer(:,2) == current,3);
            labels_list = [labels_list,label_Min];
            seq_no_Min = IndexMinLayer(IndexMinLayer(:,3)==label_Min,2);
            if size(seq_no_Min,2)~=1
                bug = 1
            end
            seq_no = [seq_no; seq_no_Min];
            todo = [todo; seq_no_Min];
        end
        todo = unique(todo);
        for j = 1:size(currents,1)
            todo(todo==currents(j)) = [];
        end
        if size(todo,2)==0
            break
        end
        
        currents = todo;
        for j = 1:size(currents,1)
            current = currents(j);
            label_Max = IndexMaxLayer(IndexMaxLayer(:,2) == current,3);
            labels_list = [labels_list,label_Max];
            seq_no_Max = IndexMaxLayer(IndexMaxLayer(:,3)==label_Max,2);
            seq_no = [seq_no;seq_no_Max];
            todo = [todo; seq_no_Max];
            
        end
        todo = unique(todo);
        for j = 1:size(currents,1)
            todo(todo==currents(j)) = [];
        end
        if size(todo,2)==0
            break
        end
        
    end
    seq_no = unique(seq_no);
    labels_list = unique(labels_list);
    cc(i).label = labels_list;
    cc(i).seq_no = seq_no;
end


