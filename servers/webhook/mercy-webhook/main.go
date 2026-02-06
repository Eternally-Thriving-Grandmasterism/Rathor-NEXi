// servers/webhook/mercy-webhook/main.go – Mercy Admission Webhook Server v1
// Blocks pod eviction during spot interruption / consolidation if valence-critical
// MIT License – Autonomicity Games Inc. 2026

package main

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"

	admissionv1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
)

var (
	scheme = runtime.NewScheme()
	codecs = serializer.NewCodecFactory(scheme)
)

func init() {
	_ = corev1.AddToScheme(scheme)
	_ = admissionv1.AddToScheme(scheme)
}

type webhookServer struct {
	certFile string
	keyFile  string
}

func (ws *webhookServer) serve(w http.ResponseWriter, r *http.Request) {
	var body []byte
	if r.Body != nil {
		if data, err := io.ReadAll(r.Body); err == nil {
			body = data
		}
	}

	if len(body) == 0 {
		http.Error(w, "empty body", http.StatusBadRequest)
		return
	}

	req := admissionv1.AdmissionReview{}
	if _, _, err := codecs.UniversalDeserializer().Decode(body, nil, &req); err != nil {
		log.Printf("Failed to decode admission review: %v", err)
		http.Error(w, "could not decode", http.StatusBadRequest)
		return
	}

	resp := ws.mutate(&req)
	respBytes, err := json.Marshal(resp)
	if err != nil {
		log.Printf("Failed to marshal response: %v", err)
		http.Error(w, "could not marshal", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write(respBytes)
}

func (ws *webhookServer) mutate(ar *admissionv1.AdmissionReview) *admissionv1.AdmissionReview {
	req := ar.Request
	resp := &admissionv1.AdmissionReview{
		TypeMeta: metav1.TypeMeta{
			APIVersion: admissionv1.SchemeGroupVersion.String(),
			Kind:       "AdmissionReview",
		},
		Response: &admissionv1.AdmissionResponse{
			UID:     req.UID,
			Allowed: true,
		},
	}

	if req.Operation != admissionv1.Delete || req.Kind.Kind != "Pod" {
		return resp
	}

	pod := &corev1.Pod{}
	if err := json.Unmarshal(req.Object.Raw, pod); err != nil {
		resp.Response.Allowed = false
		resp.Response.Result = &metav1.Status{
			Message: fmt.Sprintf("Failed to unmarshal pod: %v", err),
		}
		return resp
	}

	// Mercy gate: protect high-priority pods during disruption
	if pod.Labels["mercy-priority"] == "high" {
		// In real impl: query currentValence projection from valence-tracker service
		// Here we simulate: assume protected if label present
		log.Printf("[MercyWebhook] Blocked eviction of high-priority pod: %s/%s", pod.Namespace, pod.Name)
		resp.Response.Allowed = false
		resp.Response.Result = &metav1.Status{
			Message: "Mercy gate blocked eviction – high-valence inference pod",
			Code:    http.StatusForbidden,
		}
	}

	return resp
}

func main() {
	var certDir string
	flag.StringVar(&certDir, "cert-dir", "/etc/webhook/certs", "Directory containing tls.crt & tls.key")
	flag.Parse()

	certFile := filepath.Join(certDir, "tls.crt")
	keyFile := filepath.Join(certDir, "tls.key")

	if _, err := os.Stat(certFile); os.IsNotExist(err) {
		log.Fatalf("Cert file not found: %s", certFile)
	}
	if _, err := os.Stat(keyFile); os.IsNotExist(err) {
		log.Fatalf("Key file not found: %s", keyFile)
	}

	cert, err := tls.LoadX509KeyPair(certFile, keyFile)
	if err != nil {
		log.Fatalf("Failed to load cert/key: %v", err)
	}

	server := &webhookServer{
		certFile: certFile,
		keyFile:  keyFile,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/mutate", server.serve)
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	srv := &http.Server{
		Addr:    ":8443",
		Handler: mux,
		TLSConfig: &tls.Config{
			Certificates: []tls.Certificate{cert},
		},
	}

	log.Printf("Starting Mercy Admission Webhook on :8443")
	if err := srv.ListenAndServeTLS("", ""); err != nil {
		log.Fatalf("Failed to serve: %v", err)
	}
}
