#include <ai.h>
#include <cstring>
#include <map>

#define _CRT_SECURE_NO_WARNINGS

// A Simple Shader
AI_SHADER_NODE_EXPORT_METHODS(SimpleMethods);


struct BrdfData_wrap
{
	void* brdf_data;
	AtShaderGlobals* sg;
	float eta;
	AtVector V;
	AtVector N;
	mutable float kr;
};


struct ShaderData
{
	AtSampler* diffuse_sampler;
	AtSampler* glossy_sampler;
	AtSampler* refraction_sampler;
	int GI_diffuse_depth;
	int GI_reflection_depth;
	int GI_glossy_depth;
	int GI_diffuse_samples;
	int GI_glossy_samples;
	int diffuse_sample_offset;
	int glossy_sample_offset;
	int total_samples;
	AtCritSec cs;

	// AOV names
	std::string aov_diffuse_color;
	std::string aov_direct_diffuse;
	std::string aov_direct_diffuse_raw;
	std::string aov_indirect_diffuse;
	std::string aov_indirect_diffuse_raw;
	std::string aov_direct_specular;
	std::string aov_indirect_specular;
};

namespace
{
	enum SimpleParams { 
		p_basecolor, 
		p_roughness,
		p_ior,
		p_aov_diffuse_color,
		p_aov_direct_diffuse,
		p_aov_direct_diffuse_raw,
		p_aov_indirect_diffuse,
		p_aov_indirect_diffuse_raw,
		p_aov_direct_specular,
		p_aov_indirect_specular,
	};
};

node_parameters
{
	AiParameterRGB("baseColor", 0.5f, 0.5f, 0.5f);
	AiParameterFlt("roughness", 0.5f);
	AiParameterFlt("ior", 1.4f);
	AiParameterStr("aov_diffuse_color", "diffuse_color");
	AiParameterStr("aov_direct_diffuse", "direct_diffuse");
	AiParameterStr("aov_direct_diffuse_raw", "direct_diffuse_raw");
	AiParameterStr("aov_indirect_diffuse", "indirect_diffuse");
	AiParameterStr("aov_indirect_diffuse_raw", "indirect_diffuse_raw");
	AiParameterStr("aov_direct_specular", "direct_specular");
	AiParameterStr("aov_indirect_specular", "indirect_specular");
}

node_initialize
{
	ShaderData* data = new ShaderData;
	AiNodeSetLocalData(node, data);
	data->glossy_sampler = NULL;
}

node_update
{
	ShaderData *data = (ShaderData*)AiNodeGetLocalData(node);
	data->aov_diffuse_color = params[p_aov_diffuse_color].STR;
	data->aov_direct_diffuse = params[p_aov_direct_diffuse].STR;
	data->aov_direct_diffuse_raw = params[p_aov_direct_diffuse_raw].STR;
	data->aov_indirect_diffuse = params[p_aov_indirect_diffuse].STR;
	data->aov_indirect_diffuse_raw = params[p_aov_indirect_diffuse_raw].STR;
	data->aov_direct_specular = params[p_aov_direct_specular].STR;
	data->aov_indirect_specular = params[p_aov_indirect_specular].STR;

	AiAOVRegister(data->aov_diffuse_color.c_str(), AI_TYPE_RGB, AI_AOV_BLEND_OPACITY);
	AiAOVRegister(data->aov_direct_diffuse.c_str(), AI_TYPE_RGB, AI_AOV_BLEND_OPACITY);
	AiAOVRegister(data->aov_direct_diffuse_raw.c_str(), AI_TYPE_RGB, AI_AOV_BLEND_OPACITY);
	AiAOVRegister(data->aov_indirect_diffuse.c_str(), AI_TYPE_RGB, AI_AOV_BLEND_OPACITY);
	AiAOVRegister(data->aov_indirect_diffuse_raw.c_str(), AI_TYPE_RGB, AI_AOV_BLEND_OPACITY);
	AiAOVRegister(data->aov_direct_specular.c_str(), AI_TYPE_RGB, AI_AOV_BLEND_OPACITY);
	AiAOVRegister(data->aov_indirect_specular.c_str(), AI_TYPE_RGB, AI_AOV_BLEND_OPACITY);

	// store some options we'll reuse later
	AtNode *options = AiUniverseGetOptions();
	data->GI_glossy_samples = AiNodeGetInt(options, "GI_glossy_samples"); 
	//data->GI_glossy_samples = 8;

	// setup samples
	AiSamplerDestroy(data->glossy_sampler);
	data->glossy_sampler = AiSampler(data->GI_glossy_samples, 2);
}

node_finish
{
	if (AiNodeGetLocalData(node))
	{
		ShaderData* data = (ShaderData*)AiNodeGetLocalData(node);
		AiSamplerDestroy(data->glossy_sampler);
		AiNodeSetLocalData(node, NULL);
		delete data;
	}
}

// based on Brian Karis' SIGGRAPH 2013 talk
void importanceSampleGGX(const AtVector2 sample, float roughness, AtVector N, AtVector &outH)
{
	float alpha2 = roughness * roughness;

	float phi = 2.f * AI_PI * sample.x;
	float cosTheta = sqrtf((1.f - sample.y) / (1.f + (alpha2 * alpha2 - 1.f) * sample.y));
	float sinTheta = sqrtf(1.f - cosTheta * cosTheta);

	AtVector H = AiVector((sinTheta * cosf(phi)), (sinTheta * sinf(phi)), cosTheta);

	AtVector upVector = abs(N.z) < 0.999f ? AiVector(0.f, 0.f, 1.f) : AiVector(1.f, 0.f, 0.f);
	AtVector tangentX = AiV3Normalize(AiV3Cross(upVector, N));
	AtVector tangentY = AiV3Cross(N, tangentX);

	float Hx, Hy, Hz;
	Hx = H.x;
	Hy = H.y;
	Hz = H.z;
	outH = (tangentX * AiVector(Hx, Hx, Hx)) + (tangentY * AiVector(Hy, Hy, Hy)) + (N * AiVector(Hz, Hz, Hz));
}

void G_Schlick(float NoV, float NoL, float roughness, bool isImageBasedLight, float& G)
{
	if (!isImageBasedLight)
	{
		// disney modification to reduce hotness
		roughness = (roughness + 1.f) / 2.f;
	}
	const float k = roughness * roughness / 2.f;
	const float gv = NoV / (NoV * (1.f - k) + k);
	const float gl = NoL / (NoL * (1.f - k) + k);
	G = gl * gv;
}

float fresnel(float cosi, float etai)
{
	if (cosi >= 1.0f) return 0.0f;
	float sint = etai * sqrtf(1.0f - cosi*cosi);
	if (sint >= 1.0f) return 1.0f;

	float cost = sqrtf(1.0f - sint*sint);
	float pl = (cosi - (etai * cost))
		/ (cosi + (etai * cost));
	float pp = ((etai * cosi) - cost)
		/ ((etai * cosi) + cost);
	return (pl*pl + pp*pp)*0.5f;
}

AtRGB AiCookTorranceMISBRDF_wrap(const void* brdf_data, const AtVector* indir)
{
	AtVector H;
	const BrdfData_wrap* brdfw = reinterpret_cast<const BrdfData_wrap*>(brdf_data);
	H = AiV3Normalize((*indir) + brdfw->V);
	brdfw->kr = fresnel(MAX(0.0f, AiV3Dot(H, *indir)), brdfw->eta);
	return brdfw->kr *  AiCookTorranceMISBRDF(brdfw->brdf_data, indir);
	//return AiColorCreate(brdfw->kr, brdfw->kr, brdfw->kr);
	//return AiColorCreate(H.x, H.y, H.z);
}

float AiCookTorranceMISPDF_wrap(const void* brdf_data, const AtVector* indir)
{
	const BrdfData_wrap* brdfw = reinterpret_cast<const BrdfData_wrap*>(brdf_data);
	return AiCookTorranceMISPDF(brdfw->brdf_data, indir);
}

AtVector AiCookTorranceMISSample_wrap(const void* brdf_data, float randx, float randy)
{
	const BrdfData_wrap* brdfw = reinterpret_cast<const BrdfData_wrap*>(brdf_data);
	return AiCookTorranceMISSample(brdfw->brdf_data, randx, randy);
}

void computeSpecular(const float s_roughness, const float ior, AtColor &s_direct_result, AtColor &s_indirect_result, const AtVector u, const AtVector v, AtShaderGlobals *sg)
{
	AiLightsPrepare(sg);

	//void*			d_brdf_data = AiCookTorranceMISCreateData(sg, &x, &x, s_roughness, s_roughness);
	sg->N = sg->Nf;
	void* mis;
	mis = AiCookTorranceMISCreateData(sg, &u, &v, s_roughness, s_roughness);
	BrdfData_wrap brdfw;
	brdfw.brdf_data = mis;
	brdfw.sg = sg;
	brdfw.eta = 1.0f / ior;
	brdfw.V = -sg->Rd;
	brdfw.N = sg->Nf;
	brdfw.kr = 0.0f;

	while (AiLightsGetSample(sg))
	{
		s_direct_result += AiEvaluateLightSample(sg, &brdfw, AiCookTorranceMISSample_wrap, AiCookTorranceMISBRDF_wrap, AiCookTorranceMISPDF_wrap) * AiLightGetSpecular(sg->Lp);
	}
	s_indirect_result = brdfw.kr * AiCookTorranceIntegrate(&sg->N, sg, &u, &v, s_roughness, s_roughness);
}

// based on kettle shaders
void Kettle_diffuse(const float d_roughness, AtColor &d_direct_result, AtColor &d_indirect_result, AtShaderGlobals *sg)
{
	// -------------
	// do light loop
	// -------------

	void*			d_brdf_data = AiOrenNayarMISCreateData(sg, d_roughness);

	while (AiLightsGetSample(sg))
	{
		d_direct_result += AiEvaluateLightSample(sg, d_brdf_data, AiOrenNayarMISSample, AiOrenNayarMISBRDF, AiOrenNayarMISPDF) * AiLightGetDiffuse(sg->Lp);
	}

	// indirect diffuse

	d_indirect_result = AiOrenNayarIntegrate(&sg->Nf, sg, d_roughness);
};

void doLightLoop(const AtColor Kd,
				 const float roughness, 
				 const float ior,
				 AtColor& diffuse_direct, 
				 AtColor& diffuse_indirect, 
				 AtColor& specular_direct,
				 AtColor& specular_indirect,
				 AtColor& o_diffuse,
				 AtColor& o_specular,
				 AtShaderGlobals *sg)
{
	// view vector
	AiLightsPrepare(sg);
	AtVector E = -sg->Rd;
	AtVector N = sg->N;
	AtVector L = sg->Ld;
	

	// build a local frame for sampling
	AtVector U, V;
	AiBuildLocalFramePolar(&U, &V, &sg->N);

	// LIGHT LOOP
	Kettle_diffuse(roughness, diffuse_direct, diffuse_indirect, sg);

	// DIRECT SPECULAR REFLECTION
	computeSpecular(roughness, ior, specular_direct, specular_indirect, U, V, sg);

	// compute final result
	o_diffuse = Kd * (diffuse_direct + diffuse_indirect);
	o_specular = specular_direct + specular_indirect;
	
	// FRESNEL - at the moment it's N x V
	//Fr = AiFresnelWeight(sg->N, sg->Rd, 0.04f);

	// let's try computing Fresnel using my custom GGX distribution

	//AtVector H = AiV3Normalize(L + V);
	//Fr = fresnel(MAX(0.0f, AiV3Dot(H, L)), 1.6f);
	
	/*
	AtSamplerIterator* it = AiSamplerIterator(data->glossy_sampler, sg);
	while (AiSamplerGetSample(it, sample2D))
	{

	//const float NoL = CLAMP(L.z,0.f,1.f);
	importanceSampleGGX(AiVector2(sample2D[0], sample2D[1]), roughness, N, H);
	//L = 2 * AiV3Dot(V, H) * H - V;
	const float NoL = CLAMP(AiV3Dot(N, L), 0.f, 1.f);
	const float NoV = CLAMP(AiV3Dot(N, V), 0.001f, 1.f);
	const float VoH = CLAMP(AiV3Dot(V, H), 0.f, 1.f);
	const float NoH = CLAMP(AiV3Dot(N, H), 0.001f, 1.f);
	if (NoL > 0)
	{

	G_Schlick(NoV, NoL, roughness, false, G);
	const float Fc = powf(1.f - VoH, 5.f); // fresnel
	//outF += Fc;
	outF += Fc * G * VoH / MAX(NoH * NoV, 0.001f);
	}
	samples++;
	}
	fresnel = outF / (float)samples;
	//fresnel = 0.04f + (1 - fresnel);
	//sg->out.RGB = AiColor(outF);
	}
	*/
}

shader_evaluate
{
	// get shader data
	ShaderData *data = (ShaderData*)AiNodeGetLocalData(node);

	// read parametres
	AtColor Kd = AiShaderEvalParamRGB(p_basecolor);
	float roughness = AiShaderEvalParamFlt(p_roughness);
	float ior = MAX(1.001f, AiShaderEvalParamFlt(p_ior));
	roughness = roughness * roughness;
	roughness = MAX(0.000001f, roughness);
	// no need to clamp roughness since maya slider has min and max
	//clamp roughness to (0, 1)
	//roughness = roughness > 1.f ? 1.f : (roughness < 0.f ? 0.f : roughness);

	// DIFFUSE //
	AtColor diffuse_direct = AI_RGB_BLACK;
	AtColor diffuse_indirect = AI_RGB_BLACK;
	AtColor specular_direct = AI_RGB_BLACK;
	AtColor specular_indirect = AI_RGB_BLACK;
	AtColor o_diffuse = AI_RGB_BLACK;
	AtColor o_specular = AI_RGB_BLACK;

	doLightLoop(Kd, roughness, ior, diffuse_direct, diffuse_indirect, specular_direct, specular_indirect, o_diffuse, o_specular, sg);

	// WRITE AOVs
	
	if (Kd != AI_RGB_BLACK) AiAOVSetRGB(sg, data->aov_diffuse_color.c_str(), Kd);
	if (diffuse_direct != AI_RGB_BLACK) AiAOVSetRGB(sg, data->aov_direct_diffuse.c_str(), diffuse_direct);
	if (specular_direct != AI_RGB_BLACK) AiAOVSetRGB(sg, data->aov_direct_specular.c_str(), specular_direct);
	if (diffuse_indirect != AI_RGB_BLACK) AiAOVSetRGB(sg, data->aov_indirect_diffuse.c_str(), diffuse_indirect);
	if (specular_indirect != AI_RGB_BLACK) AiAOVSetRGB(sg, data->aov_indirect_specular.c_str(), specular_indirect);
	
	// no energy conservation at the moment?
	//sg->out.RGB = o_diffuse + o_specular;
	sg->out.RGB = o_diffuse + o_specular;
	/////@@@ TO DO
	////	 Indirect specular V
	////	 AOVs support V
	////	 
	////	 Better Fresnel
	////	 Stage 1 - Build sample set V
	////	 Stage 2 - Construct microfacet H vector
	////	 Stage 3 - Schlick approximation to Fresnel for dielectrics

	////	 Specular colour parameter (metallic?)
	////	 Normal map support
	////	
}

node_loader
{
	if (i > 0)
	return false;

	node->methods = SimpleMethods;
	node->output_type = AI_TYPE_RGB;
	node->name = "simple";
	node->node_type = AI_NODE_SHADER;
	strcpy(node->version, AI_VERSION);
	return true;
}