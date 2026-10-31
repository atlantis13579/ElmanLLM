// gather scalar implementation.
int flat=i;int source=0;int selected=IB(i);int valid=selected>=0&&selected<P(11+P(64));
for(int axis=P(1)-1;axis>=0;--axis){int coordinate=flat%P(2+axis);flat/=P(2+axis);if(axis==P(64))coordinate=selected;source+=coordinate*P(19+axis);}
if(valid!=0){COPY_A(i,source);}else{W(i,0.0f);}
