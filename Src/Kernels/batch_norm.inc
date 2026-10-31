// batch_norm scalar implementation.
int channels=P(12);int spatial=P(20);int channel=(i/spatial)%channels;
W(i,(A(i)-B(channel))/EL_SQRT(C(channel)+F(64)));
