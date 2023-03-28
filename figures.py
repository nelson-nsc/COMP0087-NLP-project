import matplotlib.pyplot as plt

# gpt-2 long snippets
results = [26.186927064720408,
 24.612248807707598,
 23.003873108134243,
 22.42810483157713,
 19.634138815098698,
 18.773096348891922,
 18.562921873848442,
 23.19419074727803]

lengths = [0, 10, 20, 30, 40, 50, 60, 70]

plt.plot(lengths[:-1], results[:-1])
plt.ylabel('BLEU score')
plt.xlabel('minimum number of characters in snippet')
plt.title('GPT-2 and long snippets')

plt.show()


# marian long snippets
results = [37.81887503587737,
 37.409753269504954,
 35.711178050842165,
 34.08758073112004,
 30.829485944845583,
 31.40210217757424,
 29.67194174556461,
 31.25947692686221]

lengths = [0, 10, 20, 30, 40, 50, 60, 70]

plt.plot(lengths[:-1], results[:-1])
plt.ylabel('BLEU score')
plt.xlabel('minimum number of characters in snippet')
plt.title('Marian and long snippets')

plt.show()

# tempurature vs bleu 
results = [21.235075548792068,
 21.418083550365786,
 20.811302175015783,
 20.67548071714807,
 20.407572897606386,
 19.587856746659853]

temps = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]

fig = plt.figure(figsize=(9,5.5))
plt.plot(temps, results)
plt.xlabel('tempurature value')
plt.ylabel('BLEU')
plt.title('Tempurature parameters vs BLEU')

plt.show()