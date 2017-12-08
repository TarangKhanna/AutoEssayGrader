import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud

# text = open(essay_path).read()
# example essay
text = 'There was a new restaurant opening in town! He really wanted to go so his family went. It had opened @NUM1 minutes ago. They didn’t think there was going to be a line. They got there and they didn’t see any cars in the drive thru. They went through the drive through which started at the back of the restaurant. Then they say about @NUM2 cars. He was really mad. They were about to leave and then there were @NUM3 cars behind them, so they stayed in line. Since it was a new restraunt everyone was picking their food very slowely. They where also making the food very slowely. Then people started yelling, “@CAPS1 up,” they where also honking their horns but he was the only who wasn’t honking his horn. They where waiting for @NUM2 minutes and @NUM1 had gone @CAPS3. He was starving and he still was being very patient. He usually is not patient. He was very excited to eat at the new restraunt.  When it was his turn the cashier said, “@CAPS2 you for being very patient.” @CAPS3 the time they got their food it took @NUM6 hour. The food wasn’t even good! He showed a lot of patience that day. He didn’t honk his car horn and he wasn’t yelling. He had stayed in the car @NUM6 hour and he was starving. It is a great thing to have patience.'
wordcloud = WordCloud(height=600,width=600).generate(text)
image = wordcloud.to_image()
image.show()
print(type(image))
print(sizeof(image))
# Display the generated image:
# the matplotlib way:
#plt.imshow(wordcloud, interpolation='bilinear')
#plt.axis("off")
#plt.show()

# image = wordcloud.to_image()
# image.show()