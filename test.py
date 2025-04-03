from transformers import AutoTokenizer, AutoModel
import torch


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings, we could use pyvi, underthesea, RDRSegment to segment words
sentences = ['Đại tướng Võ Nguyên Giáp sinh ở đâu?', 
             "Ai là Võ Nguyên Giáp? ",
             'Võ Nguyên Giáp đã đóng vai trò vô cùng quan trọng trong cuộc kháng chiến chống Mỹ cứu nước từ những năm 1960 đến 1975. Ông sinh ở Quảng Bình, ông không chỉ tham gia tích cực vào các hoạt động chiến đấu, mà còn là những nhân tố quyết định trong việc bảo đảm hậu cần, sản xuất vũ khí và thuốc men, cũng như chăm sóc gia đình và xã hội trong bối cảnh chiến tranh. Nhiều phụ nữ đã tham gia lực lượng vũ trang, đảm nhận các cương vị lãnh đạo và đóng góp đáng kể trong các chiến dịch lớn, như cuộc tổng tấn công Tết Mậu Thân năm 1968. Bên cạnh đó, họ còn giữ gìn văn hóa, truyền thống và tinh thần yêu nước, góp phần động viên tinh thần chiến đấu của toàn dân. Sự hy sinh, kiên cường và sáng tạo của phụ nữ Việt Nam trong cuộc kháng chiến đã để lại dấu ấn sâu sắc trong lịch sử dân tộc và là minh chứng cho tinh thần bất khuất của người Việt Nam.']

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder')
model = AutoModel.from_pretrained('bkai-foundation-models/vietnamese-bi-encoder')

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling. In this case, mean pooling.
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

print("Sentence embeddings:")
print(sentence_embeddings)

import torch.nn.functional as F

# Compute cosine similarity between the two sentence embeddings
cosine_sim = F.cosine_similarity(sentence_embeddings[0].unsqueeze(0), sentence_embeddings[1].unsqueeze(0))

print("Cosine Similarity:", cosine_sim.item())