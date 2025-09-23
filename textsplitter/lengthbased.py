from langchain.text_splitter import CharacterTextSplitter


text="""When I was a child, I was frozen. I barely spoke and was consumed with anxiety. There was so much in me that felt locked up, that for reasons I did not know, I could not express. When I tried, it was as though my words got jumbled up, and what was reflected made it clear that my message was not received. I decided it was easier to remain silent.
From the outside in, it appeared as though I was not smart, I did not have opinions or I simply did not care. But in fact, the more I remained silent, the more I keenly observed and deeply contemplated my experiences. As I began to perceive the world very differently from those around me, I lost faith that I would ever be understood.

“I’ve felt misunderstood my whole life. But this is not just my experience. We are all deeply misunderstood.“

When I was in elementary school I was quiet and well-behaved, so I was virtually invisible. As a teacher now, I get this. We have no choice sometimes but to give our attention to the kids that are demanding it the most. I didn’t realize it at the time, but the loud and disruptive kids that seemed so free to express were misunderstood like me. They were just trying harder to be heard.

At age 12, I was sent to an all-girls Catholic school run by a nun that had been cloistered for her adult life. This wasn’t just an ordinary school. It was like entering a time warp back into the early 1900s. Everything was controlled; from the color of the clips I put in my hair, to the precise position I had to sit in.

My first day I learned that no matter how “good” I was, I would not be seen that way. An innocent misunderstanding in class lead to me being demeaned by the “Mistress of Discipline”, and I bawled uncontrollably in front of my peers that didn’t yet know me. Humiliation happens when we shame instead of seeking to understand.

I went more and more into myself. It did not feel safe to share my innermost thoughts and feelings or to tell the truth, even when I knew I’d done nothing wrong.

I survived by learning to play the game. I observed the way my words and actions were perceived and adjusted my behavior to avoid getting “in trouble”. Like most of us do, I molded myself into everyone else’s expectations of me. I became a reflection of how the world saw me and little by little I lost touch with who I was. I even began to misunderstand myself."""
text_splitter=CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    separator=" ")
result=text_splitter.split_text(text)
print(result)
