// mse_loss scalar implementation.
int count=EL_COUNT(10,11);float total=0.0f;
for(int j=0;j<count;++j){float difference=A(j)-B(j);total+=difference*difference;}
W(i,total/float(count));
