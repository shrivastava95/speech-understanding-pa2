mkdir data/clean
mv train_audio.tar data/clean/
mv valid_audio.tar data/clean/
mv testkn_audio.tar data/clean/
mv testunk_audio.tar data/clean/
mv transcripts_n2w.tar data/clean/

mkdir data/noisy
mv data/clean/testkn_audio.tar data/noisy/testkn_audio.tar
mv data/clean/testunk_audio.tar data/noisy/testunk_audio.tar
mv data/clean/transcripts_n2w.tar data/noisy/transcripts_n2w.tar
