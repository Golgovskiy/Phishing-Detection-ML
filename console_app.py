import requests
from bs4 import BeautifulSoup
import tldextract
import re
import numpy as np
from urllib.parse import urlparse, urljoin
from os.path import join as join_path
from joblib import load as load_model
from threading import Thread
from queue import Queue
from argparse import ArgumentParser

class PhishingDetector:
    
    def __init__(self, model_path= "models/"):
        
        self.model = load_model(join_path(model_path,'model_features_c.chkpt'))
        self.model_text = load_model(join_path(model_path,'text_model_c.chkpt'))
        self.vectorizer = load_model(join_path(model_path,'vectorizer_c.chkpt'))

        self.__brand_names = [
        'paypal', 'apple', 'google', 'amazon', 'facebook', 'microsoft', 'yahoo', 'ebay', 
        'linkedin', 'netflix', 'dropbox', 'chase', 'bankofamerica', 'wellsfargo', 'citibank',
        'instagram', 'twitter', 'github', 'whatsapp', 'wechat', 'snapchat', 'outlook', 
        'hotmail', 'adobe', 'spotify', 'uber', 'lyft', 'airbnb', 'hulu', 'pinterest',
        'tumblr', 'tiktok', 'reddit', 'flickr', 'imdb', 'quora', 'medium', 'payscale', 
        'glassdoor', 'venmo', 'zelle', 'pnc', 'hsbc', 'usbank', 'barclays', 'santander', 
        'bittrex', 'coinbase', 'binance', 'kraken', 'blockchain', 'bitfinex', 'bitstamp', 
        'gemini', 'cryptopia', 'litecoin', 'ripple', 'stellar', 'monero', 'dash', 'ethereum',
        'vodafone', 'verizon', 'at&t', 'tmobile', 'sprint', 'orange', 'telekom', 'vodacom', 
        'nokia', 'samsung', 'lg', 'motorola', 'oneplus', 'oppo', 'vivo', 'xiaomi', 'huawei', 
        'alibaba', 'jd', 'rakuten', 'mercadolibre', 'shopify', 'walmart', 'target', 'bestbuy', 
        'costco', 'ikea', 'kroger', 'sears', 'macy\'s', "home%20depot", 'lowe\'s', 'cvs', 'walgreens',
        'rite%20aid', 'aldi', 'lidl', 'tesco', 'carrefour', 'asda', 'sainsbury', 'marksandspencer',
        'starbucks', 'mcdonalds', 'burger%20king', 'kfc', 'taco%20bell', 'wendy\'s', 'subway', 
        'dominos', 'pizzahut', 'dunkin', 'chipotle', 'panera', 'fiveguys', 'arbys', 'wingstop',
        'popeyes', 'chickfila', 'in-n-out', 'zomato', 'doordash', 'grubhub', 'uber%20eats', 
        'postmates', 'instacart', 'hello%20fresh', 'blue%20apron', 'wines', 'drizly', 'bevmo',
        'abc', 'bbc', 'cnn', 'fox', 'nbc', 'cbs', 'sky', 'reuters', 'bloomberg', 'nytimes',
        'washington%20post', 'the%20guardian', 'forbes', 'wsj', 'financial%20times', 'the%20economist',
        'new%20yorker', 'time', 'usa%20today', 'buzzfeed', 'huffpost', 'vice', 'techcrunch',
        'gizmodo', 'wired', 'mashable', 'the%20verge', 'ars%20technica', 'engadget', 'cnet',
        'digital%20trends', 'pcmag', 'zdnet', 'tom\'s%20hardware', 'android%20authority', 
        'android%20central', 'appleinsider', 'macrumors', '9to5mac', 'imore', 'windows%20central'
        ]

        self.__model_acc = {"url_model": 0.973847, "html_model": 0.95876 }

    def __get_num_dots(self, url):
        return url.count('.')

    def __get_subdomain_level(self, url):
        ext = tldextract.extract(url)
        return len(ext.subdomain.split('.'))

    def __get_path_level(self, url):
        path = url.split('://')[-1].split('/', 1)[-1]
        return path.count('/')

    def __get_url_length(self, url):
        return len(url)

    def __get_num_dash(self, url):
        return url.count('-')

    def __get_num_dash_in_hostname(self, url):
        hostname = url.split('://')[-1].split('/')[0]
        return hostname.count('-')

    def __has_at_symbol(self, url):
        return '@' in url

    def __has_tilde_symbol(self, url):
        return '~' in url

    def __get_num_underscore(self, url):
        return url.count('_')

    def __get_num_percent(self, url):
        return url.count('%')

    def __get_num_query_components(self, url):
        query = url.split('?', 1)[-1]
        return query.count('&') + 1 if '?' in url else 0

    def __get_num_ampersand(self, url):
        return url.count('&')

    def __get_num_hash(self, url):
        return url.count('#')

    def __get_num_numeric_chars(self, url):
        return sum(c.isdigit() for c in url)

    def __no_https(self, url):
        return not url.startswith('https://')

    def __has_random_string(self, url):
        return bool(re.search(r'[a-zA-Z0-9]{10,}', url))

    def __has_ip_address(self, url):
        return bool(re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', url))

    def __domain_in_subdomains(self, url):
        ext = tldextract.extract(url)
        return ext.domain in ext.subdomain

    def __domain_in_paths(self, url):
        ext = tldextract.extract(url)
        path = url.split('://')[-1].split('/', 1)[-1]
        return ext.domain in path

    def __https_in_hostname(self, url):
        hostname = url.split('://')[-1].split('/')[0]
        return 'https' in hostname

    def __get_hostname_length(self, url):
        hostname = url.split('://')[-1].split('/')[0]
        return len(hostname)

    def __get_path_length(self, url):
        path = url.split('://')[-1].split('/', 1)[-1]
        return len(path)

    def __get_query_length(self, url):
        query = url.split('?', 1)[-1]
        return len(query) if '?' in url else 0

    def __double_slash_in_path(self, url):
        path = url.split('://')[-1].split('/', 1)[-1]
        return '//' in path

    def __get_num_sensitive_words(self, html):
        sensitive_words = ['password', 'login', 'signin', 'account', 'bank', 'secure']
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text().lower()
        return sum(word in text for word in sensitive_words)

    def __has_embedded_brand_name(self, url):
        return any(brand in url for brand in self.__brand_names)

    def __get_pct_ext_hyperlinks(self, html, domain):
        soup = BeautifulSoup(html, 'html.parser')
        links = soup.find_all('a', href=True)
        if not links:
            return 0
        ext_links = [link for link in links if domain not in link['href']]
        return len(ext_links) / len(links)

    def __get_pct_ext_resource_urls(self, html, domain):
        soup = BeautifulSoup(html, 'html.parser')
        resources = soup.find_all(['img', 'script', 'link'], src=True)
        if not resources:
            return 0
        ext_resources = [res for res in resources if domain not in res['src']]
        return len(ext_resources) / len(resources)

    def __has_insecure_forms(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        forms = soup.find_all('form')
        return any(form.get('action', '').startswith('http://') for form in forms)

    def __relative_form_action(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        forms = soup.find_all('form')
        return any(form.get('action', '').startswith('/') for form in forms)

    def __ext_form_action(self, html, domain):
        soup = BeautifulSoup(html, 'html.parser')
        forms = soup.find_all('form')
        return any(domain not in form.get('action', '') for form in forms)

    def __abnormal_form_action(self, html, domain):
        soup = BeautifulSoup(html, 'html.parser')
        forms = soup.find_all('form')
        
        # Check for forms where action does not match domain or method is not POST
        abnormal_forms = [form for form in forms if domain not in form.get('action', '') or form.get('method', '').upper() != 'POST']
        
        return int(bool(abnormal_forms))

    def __get_pct_null_self_redirect_hyperlinks(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        links = soup.find_all('a', href=True)
        if not links:
            return 0
        null_self_redirect_links = [link for link in links if link['href'] in ('#', 'javascript:void(0);', 'javascript:;')]
        return len(null_self_redirect_links) / len(links)

    def __has_frequent_domain_name_mismatch(self, url, html):
        ext = tldextract.extract(url)
        soup = BeautifulSoup(html, 'html.parser')
        links = soup.find_all('a', href=True)
        domains = [tldextract.extract(link['href']).domain for link in links]
        return domains.count(ext.domain) < len(domains) // 2

    def __has_fake_link_in_status_bar(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        links = soup.find_all('a', href=True)
        return any('onmouseover' in link.attrs for link in links)

    def __is_right_click_disabled(self, html):
        return 'event.button==2' in html

    def __has_pop_up_window(self, html):
        return 'window.open' in html

    def __submit_info_to_email(self, html):
        return 'mailto:' in html

    def __has_iframe_or_frame(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        return bool(soup.find_all(['iframe', 'frame']))

    def __is_missing_title(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        return not bool(soup.title)

    def __has_images_only_in_form(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        forms = soup.find_all('form')
        return any(not form.find_all('input', type='text') for form in forms)

    def __abnormal_ext_form_action_rt(self, html, domain):
        soup = BeautifulSoup(html, 'html.parser')
        forms = soup.find_all('form')
        
        abnormal_forms = []
        for form in forms:
            action = form.get('action', '')
            parsed_url = urlparse(action)
            
            # Check if action points to an external domain and not the same domain
            if parsed_url.netloc and parsed_url.netloc != domain:
                abnormal_forms.append(form)
        
        return int(bool(abnormal_forms))

    def __ext_meta_script_link_rt(self, html, domain):
        # Implementation for real-time external meta/script/link
        soup = BeautifulSoup(html, 'html.parser')
        meta_scripts_links = soup.find_all(['meta', 'script', 'link'])
        if not meta_scripts_links:
            return 0
        ext_meta_scripts_links = [tag for tag in meta_scripts_links if domain not in tag.get('content', '') and domain not in tag.get('src', '')]
        return len(ext_meta_scripts_links) / len(meta_scripts_links)

    def __check_external_favicon(self, url):
        try:
            # Send a GET request to fetch the webpage
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for bad response status

            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find the favicon URL in the HTML
            favicon_url = None
            for link in soup.find_all('link', rel=['icon', 'shortcut icon']):
                if link.get('href'):
                    favicon_url = link['href']
                    break
            
            if not favicon_url:
                return False  # No favicon found, might indicate phishing

            # Extract domains from main URL and favicon URL
            parsed_main_url = urlparse(url)
            parsed_favicon_url = urlparse(urljoin(url, favicon_url))

            # Compare domains
            if parsed_main_url.netloc != parsed_favicon_url.netloc:
                return True  # Favicon is loaded from an external domain

            return False  # Favicon is loaded from the same domain

        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return False

    def __extract_features(self, url):
        response = requests.get(url)
        html = response.text
        domain = tldextract.extract(url).domain

        features = {
            'NumDots':                              self.__get_num_dots(url),
            'SubdomainLevel':                       self.__get_subdomain_level(url),
            'PathLevel':                            self.__get_path_level(url),
            'UrlLength':                            self.__get_url_length(url),
            'NumDash':                              self.__get_num_dash(url),
            'NumDashInHostname':                    self.__get_num_dash_in_hostname(url),
            'AtSymbol':                             self.__has_at_symbol(url),
            'TildeSymbol':                          self.__has_tilde_symbol(url),
            'NumUnderscore':                        self.__get_num_underscore(url),
            'NumPercent':                           self.__get_num_percent(url),
            'NumQueryComponents':                   self.__get_num_query_components(url),
            'NumAmpersand':                         self.__get_num_ampersand(url),
            'NumHash':                              self.__get_num_hash(url),
            'NumNumericChars':                      self.__get_num_numeric_chars(url),
            'NoHttps':                              self.__no_https(url),
            'RandomString':                         self.__has_random_string(url),
            'IpAddress':                            self.__has_ip_address(url),
            'DomainInSubdomains':                   self.__domain_in_subdomains(url),
            'DomainInPaths':                        self.__domain_in_paths(url),
            'HttpsInHostname':                      self.__https_in_hostname(url),
            'HostnameLength':                       self.__get_hostname_length(url),
            'PathLength':                           self.__get_path_length(url),
            'QueryLength':                          self.__get_query_length(url),
            'DoubleSlashInPath':                    self.__double_slash_in_path(url),
            'NumSensitiveWords':                    self.__get_num_sensitive_words(html),
            'EmbeddedBrandName':                    self.__has_embedded_brand_name(url),
            'PctExtHyperlinks':                     self.__get_pct_ext_hyperlinks(html, domain),
            'PctExtResourceUrls':                   self.__get_pct_ext_resource_urls(html, domain),
            'ExtFavicon':                           self.__check_external_favicon(url),
            'InsecureForms':                        self.__has_insecure_forms(html),
            'RelativeFormAction':                   self.__relative_form_action(html),
            'ExtFormAction':                        self.__ext_form_action(html, domain),
            'AbnormalFormAction':                   self.__abnormal_form_action(html, domain),
            'PctNullSelfRedirectHyperlinks':        self.__get_pct_null_self_redirect_hyperlinks(html),
            'FrequentDomainNameMismatch':           self.__has_frequent_domain_name_mismatch(url, html),
            'FakeLinkInStatusBar':                  self.__has_fake_link_in_status_bar(html),
            'RightClickDisabled':                   self.__is_right_click_disabled(html),
            'PopUpWindow':                          self.__has_pop_up_window(html),
            'SubmitInfoToEmail':                    self.__submit_info_to_email(html),
            'IframeOrFrame':                        self.__has_iframe_or_frame(html),
            'MissingTitle':                         self.__is_missing_title(html),
            'ImagesOnlyInForm':                     self.__has_images_only_in_form(html),
            'AbnormalExtFormActionR':               self.__abnormal_ext_form_action_rt(html, domain),
            'ExtMetaScriptLinkRT':                  self.__ext_meta_script_link_rt(html, domain),
        }
        
        return features, html
    
    def __wrapper(self, func, arg, queue):
        queue.put(func(arg))

    def __predict_url(self, features):
        feature_values = [features[col] for col in sorted(features.keys())]
        return self.model.predict([feature_values])[0]
        
    def __predict_html(self, html):
        text = re.sub('[^A-Za-z0-9]+', ' ', str(html))
        X = self.vectorizer.transform([text])
        return self.model_text.predict(X.astype('float32'))[0]

    def predict(self, url):
        features, html = self.__extract_features(url)
        
        q1, q2 = Queue(), Queue()
        Thread(target=self.__wrapper, args=(self.__predict_html, html, q1)).start() 
        Thread(target=self.__wrapper, args=(self.__predict_url, features, q2)).start() 
        p_html, p_url = q1.get(), q2.get()
        
        weighted_p_url = p_url*self.__model_acc["url_model"]
        weighted_p_html = p_html*self.__model_acc["html_model"]
        mean = np.mean([weighted_p_url, weighted_p_html])
        return int(np.round(mean))
        
        
        
 
if __name__ == '__main__':
    parser = ArgumentParser(description='Provide URL')
    parser.add_argument('target', type=str, help='Url to check')
    args = parser.parse_args()
    detector = PhishingDetector()
    try:
        result = detector.predict(str(args.target))
        print("This website is likely phishing." if result == 1 else "This website is likely legitimate.")
    except requests.exceptions.MissingSchema:
        print("Bad url.")
    except:
        print("Error accessing website.")
    
    
