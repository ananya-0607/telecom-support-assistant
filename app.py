"""Query-only agent interface; the API owns models and storage."""
import os
import httpx
import streamlit as st

st.set_page_config(page_title='Telecom Support Assistant', page_icon='📡', layout='wide')
st.markdown('''<style>
.block-container {max-width:1100px; padding-top:2rem; padding-bottom:3rem;}
h1 {font-size:2rem !important; letter-spacing:-0.03em;}
[data-testid="stForm"] {border-radius:12px; padding:1.4rem;}
</style>''', unsafe_allow_html=True)
st.sidebar.title('Telecom Support')
st.sidebar.caption('Agent workspace')
page = st.sidebar.radio('Page', ['Resolve complaint', 'Add knowledge'])
st.title('Resolve a complaint' if page == 'Resolve complaint' else 'Add knowledge')
st.caption('Get a suggested resolution with supporting sources.' if page == 'Resolve complaint' else 'Save a reviewed conversation or upload a knowledge-base PDF.')


def show_classification(labels):
    with st.container(border=True):
        left, right = st.columns(2)
        left.markdown('**Category**  \n'+', '.join(labels['categories']))
        left.markdown('**Product**  \n'+', '.join(labels['products']))
        right.markdown('**Severity**  \n'+labels['severity'])
        right.markdown('**Sentiment**  \n'+labels['sentiment'])


@st.fragment(run_every='3s')
def show_processing(api_url, source_kind):
    job_id = st.session_state.get('ingestion_job')
    if not job_id:
        return
    if st.session_state.get('ingestion_kind') != source_kind:
        return
    job = st.session_state.get('ingestion_status', {})
    if job.get('job_id') != job_id or job.get('status') not in ['completed', 'failed', 'interrupted']:
        try:
            response = httpx.get(api_url+'/ingestion/jobs/'+job_id, timeout=15)
            if not response.is_success:
                st.error('Could not check processing status. Please try again.')
                return
            job = response.json()
            st.session_state['ingestion_status'] = job
        except httpx.HTTPError:
            st.warning('Waiting for the backend. Status will refresh automatically.')
            return
    if job['status'] == 'completed':
        message = 'Done — this PDF is unchanged.' if job.get('no_changes') else (
            'Done — the PDF has been updated and is ready to search.' if job.get('replaced_document_id') else
            'Done — your knowledge is saved and ready to search.')
        if job.get('ticket_id'):
            message += ' Ticket: '+job['ticket_id']
        st.success(message)
    elif job['status'] in ['failed', 'interrupted']:
        st.error(job.get('error', 'Processing was interrupted. Please submit again.'))
    elif job['status'] == 'awaiting_confirmation':
        st.info('Confirm the suggested labels before processing this PDF.')
        st.write('**Category:** '+job['proposed_labels']['category'])
        st.write('**Product:** '+job['proposed_labels']['product'])
        st.caption(job.get('suggestion_reason', ''))
        if st.button('Confirm labels and process PDF', type='primary'):
            try:
                response = httpx.post(api_url+'/ingestion/jobs/'+job_id+'/confirm-labels', timeout=30)
                if response.is_success:
                    st.session_state.pop('ingestion_status', None)
                    st.rerun()
                else:
                    st.error(str(response.json().get('detail', 'Could not confirm labels.')))
            except httpx.HTTPError:
                st.error('Could not reach the backend. Check progress before trying again.')
    else:
        st.info('Processing your source. You can leave this page open; progress updates automatically.')
        if job.get('chunks_total'):
            completed, total = job.get('chunks_completed', 0), job['chunks_total']
            st.progress(completed/total, text=f'PDF sections prepared: {completed} of {total}')

if page == 'Add knowledge':
    api_url = os.getenv('SUPPORT_API_URL', 'http://127.0.0.1:8000')
    kind = st.radio('Source', ['Current generated conversation', 'New PDF'])
    pdf_mode = st.radio('PDF type', ['Existing categories', 'New category or product', 'Replace existing PDF']) if kind == 'New PDF' else None
    selection = kind if pdf_mode is None else kind+' / '+pdf_mode
    replacement_id = None
    if pdf_mode == 'Replace existing PDF':
        try:
            response = httpx.get(api_url+'/ingestion/documents', timeout=15)
            if not response.is_success:
                st.error('Could not load existing documents.')
                st.stop()
            documents = response.json()
            if not documents:
                st.info('Upload a PDF before replacing one.')
                st.stop()
            selected = st.selectbox('Document to replace', documents,
                format_func=lambda d:d['name']+f" ({d['chunks']} sections)")
            replacement_id = selected['document_id']
            st.caption('Upload its updated version. Other documents remain available.')
            selection += ' / '+replacement_id
        except httpx.HTTPError:
            st.error('Could not load documents. Check that the backend is running.')
            st.stop()
    if kind == 'Current generated conversation':
        current = st.session_state.get('result')
        if not current or current['resolution']['status'] == 'insufficient_evidence' or not current['resolution']['steps']:
            st.info('First generate a supported answer on Resolve complaint, then return here to save it.')
            st.stop()
        saved_complaint = st.session_state.get('resolved_complaint', '')
        draft_steps = '\n'.join(f"{n}. {step['instruction']}" for n,step in enumerate(current['resolution']['steps'], 1))
        show_classification(current['classification'])
    with st.form('addition'):
        if kind == 'Current generated conversation':
            complaint = saved_complaint
            st.text_area('Customer complaint', value=complaint, disabled=True)
            steps = st.text_area('Review resolution steps', value=draft_steps, key='steps_'+str(hash((saved_complaint, draft_steps))))
            summary = st.text_area('Review summary', value=current['resolution']['summary'], key='summary_'+str(hash((saved_complaint, current['resolution']['summary']))))
            reviewed = st.checkbox('I reviewed these steps and approve them for reuse as support knowledge.')
        else:
            upload = st.file_uploader('Updated PDF (maximum 10 MB)' if replacement_id else 'New PDF (maximum 10 MB)', type=['pdf'])
            st.caption('Use a PDF with selectable text. Scanned documents are not supported.')
            if pdf_mode == 'New category or product':
                st.caption('Provide one or both labels. Leave a label blank to request a suggestion from the existing list. New names need descriptions.')
                category = st.text_input('Category name (optional)')
                category_description = st.text_input('Category description')
                product = st.text_input('Product name (optional)')
                product_description = st.text_input('Product description')
        submit = st.form_submit_button('Save conversation' if kind == 'Current generated conversation' else
            ('Replace PDF' if replacement_id else 'Upload PDF'), type='primary')
    if submit:
        try:
            if kind == 'Current generated conversation':
                if not reviewed:
                    st.error('Review the suggested steps and confirm before saving.')
                    st.stop()
                remaining = ''
                if current['resolution']['missing_information']:
                    remaining += '\nInformation to confirm:\n'+'\n'.join(current['resolution']['missing_information'])
                if current['resolution']['escalation']:
                    remaining += '\nEscalation: '+current['resolution']['escalation']
                conversation = 'Customer: '+complaint+'\nAgent: '+summary+'\n'+steps+remaining
                response = httpx.post(api_url+'/ingestion/conversations', json=dict(
                    customer_complaint=complaint, conversation=conversation,
                    resolution_steps=steps, resolution_summary=summary,
                    classification=current['classification'], reviewed=True,
                    generated_response=current), timeout=30)
            elif upload is None:
                st.error('Choose a PDF first.')
                st.stop()
            else:
                parameters = dict(new_labels='true', category=category,
                    category_description=category_description, product=product,
                    product_description=product_description) if pdf_mode == 'New category or product' else {}
                parameters['filename'] = upload.name
                if replacement_id:
                    parameters['replacement_id'] = replacement_id
                response = httpx.post(api_url+'/ingestion/pdfs', content=upload.getvalue(),
                    headers={'Content-Type':'application/pdf'}, params=parameters, timeout=30)
            if response.is_success:
                st.session_state['ingestion_job'] = response.json()['job_id']
                st.session_state['ingestion_kind'] = selection
                st.session_state.pop('ingestion_status', None)
            else:
                st.error(str(response.json().get('detail', 'Submission failed.')))
        except httpx.HTTPError:
            st.error('Could not reach backend. Check whether submission was accepted before resubmitting.')
    show_processing(api_url, selection)
    st.stop()
with st.form('complaint_form'):
    complaint = st.text_area('Customer complaint', height=150,
        placeholder='My broadband disconnects every evening and interrupts my work calls...')
    submitted = st.form_submit_button('Find resolution', type='primary')
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
                    st.session_state['resolved_complaint'] = complaint
                else:
                    detail = response.json().get('detail','Request failed.')
                    st.error(detail if isinstance(detail,str) else 'Invalid complaint input.')
            except httpx.HTTPError:
                st.error('Could not reach the API or the request timed out. Check the API terminal before retrying.')

if 'result' in st.session_state:
    result = st.session_state['result']
    labels = result['classification']
    st.subheader('Classification')
    show_classification(labels)
    resolution = result['resolution']
    st.subheader('Resolution')
    st.write(resolution['summary'])
    if resolution['status'] == 'insufficient_evidence':
        st.write('Please provide more details or refer this complaint for specialist review.')
    for number,step in enumerate(resolution['steps'],1):
        st.markdown(f"**{number}.** {step['instruction']}")
    if resolution['missing_information']:
        st.subheader('Information to ask')
        for item in resolution['missing_information']:
            st.write('• '+item)
    if resolution['escalation']:
        st.write('**Escalation:** '+resolution['escalation'])
    st.subheader('Citations')
    for evidence in result['sources']:
        source = evidence['source']
        description = (source['ticket_id'] if source['source_type']=='ticket' else
            f"{source['title'] or 'KB passage'} — pages {', '.join(str(p) for p in source['pages'])}")
        with st.expander(description):
            st.text(evidence['passage_text'])
            if source['source_type']=='ticket':
                st.write('**Historical resolution:**')
                st.text(source['resolution_steps'])
                st.text(source['resolution_summary'])
    st.caption('Review the suggested resolution against the cited evidence before taking action.')
