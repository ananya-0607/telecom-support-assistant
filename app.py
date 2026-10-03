"""Query-only agent interface; the API owns models and storage."""
import os
import httpx
import streamlit as st

st.set_page_config(page_title='Telecom Support Assistant', page_icon='📡', layout='wide')
st.title('Telecom Support Resolution Assistant')
st.caption('Paste a complaint to find historical evidence and draft cited resolution steps.')
with st.form('complaint_form'):
    complaint = st.text_area('Customer complaint', height=150,
        placeholder='My broadband disconnects every evening and interrupts my work calls...')
    submitted = st.form_submit_button('Get resolution', type='primary')
if submitted:
    st.session_state.pop('result',None)
    if not complaint.strip():
        st.warning('Enter a complaint first.')
    else:
        with st.spinner('Classifying, finding evidence and drafting resolution...'):
            try:
                response = httpx.post(os.getenv('SUPPORT_API_URL','http://127.0.0.1:8000')+'/resolve',
                    json={'complaint':complaint}, timeout=180)
                if response.is_success:
                    st.session_state['result'] = response.json()
                else:
                    detail = response.json().get('detail','Request failed.')
                    st.error(detail if isinstance(detail,str) else 'Invalid complaint input.')
            except httpx.HTTPError:
                st.error('Could not reach the API or the request timed out. Check the API terminal before retrying.')

if 'result' in st.session_state:
    result = st.session_state['result']
    labels = result['classification']
    st.subheader('Classification')
    left,right = st.columns(2)
    left.write('**Categories:** '+', '.join(labels['categories']))
    left.write('**Products:** '+', '.join(labels['products']))
    right.write('**Severity:** '+labels['severity'])
    right.write('**Sentiment:** '+labels['sentiment'])
    resolution = result['resolution']
    if any(item.get('warning') for item in result['retrieval']):
        st.warning('Some sources matched keywords without passing the semantic threshold. Review their relevance carefully.')
    elif any(item['scope'] != 'category_product' for item in result['retrieval']):
        st.info('Search was broadened beyond the initial labels to find additional evidence.')
    st.subheader('Suggested resolution')
    st.write(resolution['summary'])
    if resolution['status'] == 'insufficient_evidence':
        st.warning('Insufficient evidence for a supported resolution.')
    for number,step in enumerate(resolution['steps'],1):
        st.markdown(f"**{number}.** {step['instruction']} " + ' '.join(f"[{c}]" for c in step['citations']))
    if resolution['missing_information']:
        st.write('**Information to confirm:**')
        for item in resolution['missing_information']:
            st.write('• '+item)
    if resolution['escalation']:
        st.write('**Escalation:** '+resolution['escalation'])
    st.subheader('Supporting sources')
    for evidence in result['sources']:
        source = evidence['source']
        description = (source['ticket_id'] if source['source_type']=='ticket' else
            f"{source['title'] or 'KB passage'} — {source['source_file']} — pages {source['pages']}")
        with st.expander(f"[{evidence['citation_id']}] {description}"):
            st.text(evidence['passage_text'])
            if source['source_type']=='ticket':
                st.write('**Historical resolution:**')
                st.text(source['resolution_steps'])
                st.text(source['resolution_summary'])
    st.caption(result['notice'])
    with st.expander('Search details and timing'):
        st.json({'retrieval':result['retrieval'], 'timings_seconds':result['timings_seconds']})
