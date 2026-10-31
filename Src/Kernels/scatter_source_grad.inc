// scatter_source_grad scalar implementation.
int flat=i;int target=0;int selected=IB(i);int valid=selected>=0&&selected<P(11+P(64));
for(int axis=P(1)-1;axis>=0;--axis){int coordinate=flat%P(2+axis);flat/=P(2+axis);if(axis==P(64))coordinate=selected;target+=coordinate*P(19+axis);}
for(int j=i+1;j<P(0);++j){int other_flat=j;int other_target=0;int other_selected=IB(j);
    for(int axis=P(1)-1;axis>=0;--axis){int coordinate=other_flat%P(2+axis);other_flat/=P(2+axis);if(axis==P(64))coordinate=other_selected;other_target+=coordinate*P(19+axis);}
    if(other_target==target&&other_selected>=0&&other_selected<P(11+P(64)))valid=0;
}
W(i,valid!=0?A(target):0.0f);
