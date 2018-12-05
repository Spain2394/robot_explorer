a=1
for i in *.png; do
  new=$(printf "image_%01d.jpg" "$a") #04 pad to length of 4
  mv -i -- "$i" "$new"
  let a=a+1
done
