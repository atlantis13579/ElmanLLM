// scatter scalar implementation.
float value=A(i);int integer_value=IA(i);
int count=EL_COUNT(44,45);
for(int j=0;j<count;++j){int flat=j;int target=0;int selected=IB(j);
    for(int axis=P(1)-1;axis>=0;--axis){int coordinate=flat%P(45+axis);flat/=P(45+axis);if(axis==P(64))coordinate=selected;target+=coordinate*P(19+axis);}
    if(target==i&&selected>=0&&selected<P(2+P(64))){value=C(j);integer_value=IC(j);}
}
if(P(255)==0)W(i,value);else WI(i,integer_value);
